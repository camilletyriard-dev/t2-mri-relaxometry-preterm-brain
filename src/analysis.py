"""
Analysis pipelines for T2 relaxometry.

Provides cohort-level fitting loops, voxel-wise mapping, prior generation,
spatial smoothing, and statistical comparison functions.
"""

import numpy as np
import pandas as pd
from scipy import stats, ndimage

from utils import mean_decay_curve, hard_seg
from models import (mono_exp, bi_exp_fixed,
                    fit_log_linear, fit_nlls,
                    fit_bi_fixed, fit_bi_free,
                    fit_bi_fixed_map,
                    parametric_bootstrap_mono,
                    parametric_bootstrap_bi_fixed,
                    aicc, akaike_weights,
                    mcmc_bi_fixed,
                    T2_GRID, make_kernel,
                    nnls_plain, pick_mu_chi2, mwf_from_spectrum,
                    three_exp_fixed, fit_three_fixed)
from quality import estimate_sigma_bg


# ===================================================================
# Task 2: one-compartment fitting
# ===================================================================

def two_point_T2_map(reg, TE, mask, idx_a=0, idx_b=-1):
    """Analytical T2 at every in-mask voxel from two echoes."""
    T2 = np.full(reg.shape[:3], np.nan)
    Sa = reg[..., idx_a].astype(float)
    Sb = reg[..., idx_b].astype(float)
    ok = mask & (Sa > 0) & (Sb > 0) & (Sa > Sb)
    T2[ok] = (TE[idx_b] - TE[idx_a]) / np.log(Sa[ok] / Sb[ok])
    T2[(T2 <= 0) | ~np.isfinite(T2)] = np.nan
    return T2


def voxelwise_log_linear_T2(reg, TE, mask):
    """Log-linear OLS T2 map, looping over in-mask voxels."""
    T2_map = np.full(reg.shape[:3], np.nan)
    for x, y, z in np.argwhere(mask):
        _, T2 = fit_log_linear(TE, reg[x, y, z, :].astype(float))
        T2_map[x, y, z] = T2
    return T2_map


def fit_cohort_mono(preterm, tissues=None):
    """
    Fit OLS, WLS, NLLS to tissue-mean curves for every subject.

    Returns a DataFrame with columns: id, EP, tissue, T2_OLS, T2_WLS, T2_NLLS.
    """
    from models import fit_wls
    if tissues is None:
        tissues = {'CSF': 1, 'GM': 2, 'WM': 3}

    rows = []
    for ds in preterm:
        for lab, seg_idx in tissues.items():
            curve = mean_decay_curve(ds, seg_idx)
            _, T2_ols = fit_log_linear(ds['TE'], curve)
            _, T2_wls = fit_wls(ds['TE'], curve)
            _, T2_nll = fit_nlls(ds['TE'], curve)
            rows.append(dict(
                id=ds['id'], EP=ds['EP'], tissue=lab,
                T2_OLS=round(T2_ols, 2),
                T2_WLS=round(T2_wls, 2),
                T2_NLLS=round(T2_nll, 2)))
    return pd.DataFrame(rows)


# ===================================================================
# Task 3: two-compartment fitting
# ===================================================================

# Default compartment specification
COMPARTMENTS = {
    'WM':  {'seg_idx': 3, 'T2s': 20.0, 'T2l': 80.0},
    'GM':  {'seg_idx': 2, 'T2s': 20.0, 'T2l': 80.0},
    'CSF': {'seg_idx': 1, 'T2s': 20.0, 'T2l': 80.0},
}


def fit_cohort_biexp_fixed(preterm, compartments=None):
    """
    Fixed-T2 bi-exponential fit on tissue-mean curves for every subject.

    Returns a DataFrame with columns: id, EP, tissue, S0, v.
    """
    if compartments is None:
        compartments = COMPARTMENTS

    rows = []
    for ds in preterm:
        for tissue, pars in compartments.items():
            curve = mean_decay_curve(ds, pars['seg_idx'])
            S0, v = fit_bi_fixed(ds['TE'], curve, pars['T2s'], pars['T2l'],
                                 v_init=0.10)
            rows.append(dict(id=ds['id'], EP=ds['EP'], tissue=tissue,
                             S0=round(S0, 1), v=round(v, 3)))
    return pd.DataFrame(rows)


def bootstrap_cohort_biexp(preterm, compartments=None, n_boot=200):
    """
    Parametric bootstrap CIs on the fixed-T2 bi-exp v for every subject.

    Returns a DataFrame with columns: id, EP, tissue, v, CI_low, CI_high, CI_width.
    """
    if compartments is None:
        compartments = COMPARTMENTS

    rows = []
    for ds in preterm:
        for tissue, pars in compartments.items():
            curve = mean_decay_curve(ds, pars['seg_idx'])
            v_hat, lo, hi = parametric_bootstrap_bi_fixed(
                ds['TE'], curve, pars['T2s'], pars['T2l'], n_boot=n_boot)
            rows.append(dict(
                id=ds['id'], EP=ds['EP'], tissue=tissue,
                v=round(v_hat, 3),
                CI_low=round(lo, 3),
                CI_high=round(hi, 3),
                CI_width=round(hi - lo, 3)))
    return pd.DataFrame(rows)


# ===================================================================
# Task 4: model comparison
# ===================================================================

def fit_cohort_aicc_1v2(preterm, tissues=None, T2s=20.0, T2l=80.0):
    """
    AICc comparison of 1- vs 2-compartment models per subject per tissue.

    Returns a DataFrame with AICc_1, AICc_2, dAICc columns.
    """
    if tissues is None:
        tissues = {'WM': 3, 'GM': 2, 'CSF': 1}

    rows = []
    for ds in preterm:
        TE = np.asarray(ds['TE'], float)
        for tissue, seg_idx in tissues.items():
            curve = mean_decay_curve(ds, seg_idx)

            # 1-compartment
            S0_1, T2_1 = fit_nlls(TE, curve)
            res_1 = curve - mono_exp(TE, S0_1, T2_1)
            aicc_1 = aicc(res_1, n_params=3)

            # 2-compartment fixed
            S0_2, v = fit_bi_fixed(TE, curve, T2s, T2l)
            res_2 = curve - bi_exp_fixed(TE, S0_2, v, T2s, T2l)
            aicc_2 = aicc(res_2, n_params=3)

            rows.append(dict(
                id=ds['id'], EP=ds['EP'], tissue=tissue,
                T2_1=round(T2_1, 1), v_2=round(v, 3),
                AICc_1=round(aicc_1, 2),
                AICc_2=round(aicc_2, 2),
                dAICc=round(aicc_2 - aicc_1, 2)))
    return pd.DataFrame(rows)


# ===================================================================
# Task 5: multi-compartment
# ===================================================================

def fit_cohort_nnls(preterm, tissues=None):
    """
    Regularised NNLS spectra for every subject and tissue.

    Returns dict of {tissue: list of amplitude arrays}.
    """
    if tissues is None:
        tissues = {'CSF': 1, 'GM': 2, 'WM': 3}

    spectra = {t: [] for t in tissues}
    for ds in preterm:
        for t, sid in tissues.items():
            curve = mean_decay_curve(ds, sid)
            _, a, _, _ = pick_mu_chi2(ds['TE'], curve)
            spectra[t].append(a)
    return spectra


def fit_cohort_3comp(preterm):
    """
    Compare 2-comp, 3-comp (Dingwall), and NNLS MWF on WM curves.

    Returns a DataFrame with MWF_NNLS, v_2comp, MWF_3comp columns.
    """
    rows = []
    for ds in preterm:
        wm = mean_decay_curve(ds, 3)
        _, a, _, _ = pick_mu_chi2(ds['TE'], wm)
        mwf_nnls = mwf_from_spectrum(a)
        _, v_2comp = fit_bi_fixed(ds['TE'], wm, T2s=20.0, T2l=80.0)
        S0, v1, v2 = fit_three_fixed(ds['TE'], wm)
        rows.append(dict(
            id=ds['id'], EP=ds['EP'],
            MWF_NNLS=mwf_nnls, v_2comp=v_2comp,
            MWF_3comp=v1, v_IE_3comp=v2,
            v_CSF_3comp=max(0.0, 1.0 - v1 - v2)))
    return pd.DataFrame(rows)


def fit_cohort_aicc_1v2v3(preterm):
    """
    AICc comparison of 1- vs 2- vs 3-compartment on WM curves.

    Returns a DataFrame with AICc_1, AICc_2, AICc_3, dAICc_3v2 columns.
    """
    rows = []
    for ds in preterm:
        TE = np.asarray(ds['TE'], float)
        wm = mean_decay_curve(ds, 3)

        S0_1, T2_1 = fit_nlls(TE, wm)
        aicc_1 = aicc(wm - mono_exp(TE, S0_1, T2_1), n_params=3)

        S0_2, v = fit_bi_fixed(TE, wm, T2s=20.0, T2l=80.0)
        aicc_2 = aicc(wm - bi_exp_fixed(TE, S0_2, v, 20.0, 80.0), n_params=3)

        S0_3, v1, v2 = fit_three_fixed(TE, wm)
        aicc_3 = aicc(wm - three_exp_fixed(TE, S0_3, v1, v2), n_params=4)

        rows.append(dict(id=ds['id'], EP=ds['EP'],
                         AICc_1=aicc_1, AICc_2=aicc_2, AICc_3=aicc_3))
    df = pd.DataFrame(rows)
    df['dAICc_3v2'] = df['AICc_3'] - df['AICc_2']
    return df


# ===================================================================
# Task 6: priors
# ===================================================================

def compute_cohort_priors(preterm, T2s=20.0, T2l=80.0):
    """
    Compute per-tissue cohort-level MWF (v) mean and SD from the
    fixed-T2 bi-exp fit.  These serve as the prior parameters.

    Returns (mu_cohort, sigma_cohort) as pandas Series indexed by tissue.
    """
    rows = []
    for ds in preterm:
        for tissue, seg_idx in [('CSF', 1), ('GM', 2), ('WM', 3)]:
            curve = mean_decay_curve(ds, seg_idx)
            _, v = fit_bi_fixed(ds['TE'], curve, T2s, T2l, v_init=0.10)
            rows.append(dict(tissue=tissue, v=v))

    df = pd.DataFrame(rows).dropna()
    mu_cohort = df.groupby('tissue')['v'].mean()
    sigma_cohort = df.groupby('tissue')['v'].std()
    return mu_cohort, sigma_cohort


def voxel_prior_from_softseg(ds, x, y, z, mu, sigma, sigma_floor=0.02):
    """
    Generate a voxel-specific (mu_v, sigma_v) prior for v by mixing
    cohort-level tissue MWFs with the voxel's soft-seg probabilities.
    """
    probs = np.asarray(ds['seg'][x, y, z, :], float)
    total = probs.sum()
    if total < 1e-6:
        return float(mu['WM']), float(max(sigma['WM'], sigma_floor))
    probs = probs / total

    p_csf = probs[1]
    p_gm = probs[2] + probs[4] + probs[5]      # include deep GM + brainstem
    p_wm = probs[3]

    mu_v = p_csf * mu['CSF'] + p_gm * mu['GM'] + p_wm * mu['WM']
    var = (p_csf * sigma['CSF']**2 +
           p_gm * sigma['GM']**2 +
           p_wm * sigma['WM']**2)
    sigma_v = max(float(np.sqrt(max(var, 0.0))), sigma_floor)
    return float(mu_v), float(sigma_v)


def voxelwise_mwf_maps(ds, sl, mu_cohort, sigma_cohort, sigma_noise,
                       T2s=20.0, T2l=80.0):
    """
    Compute three MWF maps on one axial slice:
      (1) no prior (v_init=0.10)
      (2) warm-start (v_init from soft-seg)
      (3) MAP (Gaussian penalty)

    Returns (mwf_no_prior, mwf_warm, mwf_map) as 2-D arrays.
    """
    TE = ds['TE']
    msl = ds['mask'][:, :, sl]
    shape = msl.shape

    mwf_no_prior = np.full(shape, np.nan)
    mwf_warm = np.full(shape, np.nan)
    mwf_map = np.full(shape, np.nan)

    for x, y in np.argwhere(msl):
        sig = ds['reg'][x, y, sl, :].astype(float)
        if np.any(sig <= 0):
            continue

        mu_v, sigma_v = voxel_prior_from_softseg(
            ds, x, y, sl, mu_cohort, sigma_cohort)

        _, v1 = fit_bi_fixed(TE, sig, T2s, T2l, v_init=0.10)
        _, v2 = fit_bi_fixed(TE, sig, T2s, T2l, v_init=mu_v)
        _, v3 = fit_bi_fixed_map(TE, sig, T2s, T2l,
                                 mu_v=mu_v, sigma_prior=sigma_v,
                                 sigma_noise=sigma_noise)
        mwf_no_prior[x, y] = v1
        mwf_warm[x, y] = v2
        mwf_map[x, y] = v3

    return mwf_no_prior, mwf_warm, mwf_map


def smooth_within_tissue(vmap, tissue_mask, fwhm_voxels=2.0):
    """
    Gaussian smooth vmap only where tissue_mask is True.
    Prevents signal leaking across tissue boundaries.
    """
    sigma = fwhm_voxels / (2 * np.sqrt(2 * np.log(2)))
    vals = np.where(tissue_mask, np.nan_to_num(vmap), 0.0)
    wts = tissue_mask.astype(float)
    num = ndimage.gaussian_filter(vals, sigma=sigma)
    den = ndimage.gaussian_filter(wts, sigma=sigma)
    out = np.full(vmap.shape, np.nan)
    safe = den > 1e-6
    out[safe] = num[safe] / den[safe]
    out[~tissue_mask] = np.nan
    return out


# ===================================================================
# Task 7: inter-subject statistics
# ===================================================================

def cohens_d(a, b):
    """Pooled-SD Cohen's d."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1)
                      + (len(b) - 1) * np.var(b, ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def bootstrap_mean_diff(a, b, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI on mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = np.array([
        rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
        for _ in range(n_boot)
    ])
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def ttest_summary(a, b, n_boot=2000, seed=0):
    """Welch t-test + Cohen's d + bootstrap 95% CI on mean(a) - mean(b)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    t_stat, p = stats.ttest_ind(a, b, equal_var=False)
    lo, hi = bootstrap_mean_diff(a, b, n_boot=n_boot, seed=seed)
    return dict(
        n_a=len(a), n_b=len(b),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        diff=float(a.mean() - b.mean()),
        CI_lo=float(lo), CI_hi=float(hi),
        d=float(cohens_d(a, b)),
        t=float(t_stat), p=float(p))


def sig_marker(p, bonf_thr=None):
    """Standard significance marker: ** Bonferroni, * uncorrected."""
    if bonf_thr is not None and p < bonf_thr:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def fit_cohort_wm(preterm, T2s=20.0, T2l=80.0):
    """
    Per-subject WM mono-exp T2 and fixed-T2 bi-exp MWF.

    Returns a DataFrame with columns: id, EP, gawks, T2_WM, v_MWF.
    """
    rows = []
    for ds in preterm:
        wm_curve = mean_decay_curve(ds, seg_idx=3)
        _, T2_wm = fit_nlls(ds['TE'], wm_curve)
        _, v_mwf = fit_bi_fixed(ds['TE'], wm_curve, T2s, T2l, v_init=0.10)
        rows.append(dict(
            id=ds['id'], EP=ds['EP'],
            gawks=ds.get('gawks', np.nan),
            T2_WM=T2_wm, v_MWF=v_mwf))
    return pd.DataFrame(rows).dropna(subset=['T2_WM', 'v_MWF'])


WM_REGIONS = {1: 'PLIC', 2: 'Genu CC', 3: 'ALIC', 4: 'Splenium CC'}


def region_mean_decay(ds, region_label):
    """WM-probability-weighted mean decay curve for a par_lobe region."""
    par = ds.get('par_lobe')
    if par is None:
        return None
    region_mask = (par == region_label) & ds['mask']
    if not region_mask.any():
        return None
    weights = ds['seg'][..., 3] * region_mask
    total = float(weights.sum())
    if total < 1e-6:
        return None
    return (ds['reg'].astype(np.float64)
            * weights[..., None]).sum(axis=(0, 1, 2)) / total


def fit_regional(preterm, wm_regions=None, T2s=20.0, T2l=80.0):
    """
    Per-subject, per-region mono-exp T2 and fixed-T2 bi-exp v.

    Returns a DataFrame with columns: id, EP, gawks, region, T2, v.
    """
    if wm_regions is None:
        wm_regions = WM_REGIONS

    rows = []
    for ds in preterm:
        for lbl, rname in wm_regions.items():
            curve = region_mean_decay(ds, lbl)
            if curve is None:
                continue
            _, T2_reg = fit_nlls(ds['TE'], curve)
            _, v_reg = fit_bi_fixed(ds['TE'], curve, T2s, T2l, v_init=0.10)
            rows.append(dict(
                id=ds['id'], EP=ds['EP'],
                gawks=ds.get('gawks', np.nan),
                region=rname, T2=T2_reg, v=v_reg))
    return pd.DataFrame(rows).dropna(subset=['T2', 'v'])


def regional_group_stats(df_reg, wm_regions=None):
    """
    EP vs FT t-tests per region per metric, with Bonferroni correction.

    Returns a DataFrame with p, p_bonf, sig columns.
    """
    if wm_regions is None:
        wm_regions = WM_REGIONS

    region_stats = []
    for metric in ['T2', 'v']:
        for rname in wm_regions.values():
            sub = df_reg[df_reg['region'] == rname]
            ep = sub.loc[sub['EP'] == True, metric].values
            ft = sub.loc[sub['EP'] == False, metric].values
            if len(ep) < 3 or len(ft) < 3:
                continue
            t_stat, p = stats.ttest_ind(ep, ft, equal_var=False)
            lo, hi = bootstrap_mean_diff(ep, ft)
            region_stats.append(dict(
                metric=metric, region=rname,
                n_EP=len(ep), n_FT=len(ft),
                EP=round(ep.mean(), 3), FT=round(ft.mean(), 3),
                diff=round(ep.mean() - ft.mean(), 3),
                CI95=f'[{lo:+.3f}, {hi:+.3f}]',
                d=round(cohens_d(ep, ft), 3),
                t=round(t_stat, 3), p=round(p, 4)))

    stats_df = pd.DataFrame(region_stats)
    n_tests = len(stats_df)
    stats_df['p_bonf'] = (stats_df['p'] * n_tests).clip(upper=1.0).round(4)
    stats_df['sig'] = np.where(
        stats_df['p_bonf'] < 0.05, '**',
        np.where(stats_df['p'] < 0.05, '*', ''))
    return stats_df
