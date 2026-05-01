"""
Quality control functions for T2 relaxometry data.

Provides monotonicity checks, SNR estimation, log-linear residual scoring,
partial-volume boundary analysis, and a consolidated QA table builder.
"""

import numpy as np
import pandas as pd
from scipy import ndimage

from utils import mean_decay_curve


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------
def check_monotonicity(reg, mask, rel_tol=0):
    """
    Test whether each voxel's decay curve is strictly non-increasing.

    Parameters
    ----------
    rel_tol : float
        Allow rises up to this fraction of the previous echo signal.

    Returns
    -------
    frac_monotone : float
        Fraction of in-mask voxels with no violations.
    mean_viol : float
        Mean violation fraction across all voxels.
    viol_map : ndarray
        3-D map of violation fraction per voxel.
    """
    vox = reg[mask, :].astype(float)
    prev = np.maximum(vox[:, :-1], 1e-6)
    diffs = np.diff(vox, axis=1)

    n_up = (diffs > rel_tol * prev).sum(axis=1)
    n_steps = diffs.shape[1]

    viol_map = np.zeros(reg.shape[:3], dtype=float)
    viol_map[mask] = n_up / n_steps

    frac_monotone = float((n_up == 0).mean())
    mean_viol = float(n_up.mean() / n_steps)
    return frac_monotone, mean_viol, viol_map


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------
def estimate_sigma_bg(reg, mask):
    """
    Estimate noise sigma from background voxels at the last echo.

    Uses the Rician correction (Gudbjartsson & Patz 1995):
    for Rician magnitude noise, background mean = sigma * sqrt(pi/2).
    """
    bg_last = reg[~mask, -1].astype(float)
    if bg_last.size == 0:
        return float(np.std(reg[..., -1]))
    return float(bg_last.mean() / np.sqrt(np.pi / 2))


def snr_per_echo(ds):
    """
    Compute median voxelwise SNR at each echo time.

    Returns (snr_curve, sigma) where snr_curve has shape (n_echoes,).
    """
    sigma = estimate_sigma_bg(ds['reg'], ds['mask'])
    snr_curve = np.array([
        np.median(ds['reg'][..., k][ds['mask']].astype(float) / max(sigma, 1e-6))
        for k in range(len(ds['TE']))
    ])
    return snr_curve, sigma


# ---------------------------------------------------------------------------
# Log-linear fit residuals (mono-exponentiality test)
# ---------------------------------------------------------------------------
def fit_logline_and_residuals(signal, TE):
    """
    Fit log(signal) vs TE by OLS and return residuals.

    Returns (x, yhat_exp, residuals) where yhat_exp is on the original scale.
    """
    keep = signal > 0
    y = np.log(signal[keep])
    x = TE[keep]
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    res = y - yhat
    return x, np.exp(yhat), res


def subject_logfit_score(ds, seg_list=(1, 2, 3)):
    """
    Mean absolute log-residual across selected tissues.

    Lower score = closer to mono-exponential. Used to rank subjects by
    how well a single exponential fits their tissue-mean decay curves.
    """
    vals = []
    for seg_idx in seg_list:
        c = mean_decay_curve(ds, seg_idx)
        _, _, res = fit_logline_and_residuals(c, ds['TE'])
        vals.append(np.mean(np.abs(res)))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Partial volume at tissue boundaries
# ---------------------------------------------------------------------------
def boundary_fractions(ds, radius=1):
    """
    Fraction of each tissue's voxels that lie at a boundary (within
    `radius` voxels of a different tissue).  Higher = more partial volume.
    """
    from utils import hard_seg
    hseg = hard_seg(ds)
    struct = ndimage.iterate_structure(
        ndimage.generate_binary_structure(3, 1), radius)
    out = {}
    for sid in range(1, 6):
        roi = (hseg == sid) & ds['mask']
        if not roi.any():
            out[sid] = np.nan
            continue
        inner = ndimage.binary_erosion(roi, structure=struct)
        out[sid] = 1 - inner.sum() / roi.sum()
    return out


def representative_brain_slices(mask, n_slices=5):
    """Pick n evenly spaced axial slices across the brain extent."""
    z_valid = np.where(mask.any(axis=(0, 1)))[0]
    if len(z_valid) == 0:
        raise ValueError('Mask contains no brain voxels.')
    z0, z1 = z_valid[0], z_valid[-1]
    return np.round(np.linspace(z0, z1, n_slices)).astype(int)


# ---------------------------------------------------------------------------
# Consolidated QA table
# ---------------------------------------------------------------------------
def cohort_qa_table(preterm):
    """
    Build a per-subject QA table with monotonicity violations, log-fit
    scores, and worst-10% flags.

    Returns (df_mono, df_logfit, qa) DataFrames.
    """
    # Monotonicity
    mono_rows = []
    for ds in preterm:
        fm, mv, _ = check_monotonicity(ds['reg'], ds['mask'])
        mono_rows.append(dict(
            id=ds['id'], EP=ds['EP'],
            strictly_monotone_pct=round(fm * 100, 1),
            mean_violation_pct=round(mv * 100, 2)))
    df_mono = pd.DataFrame(mono_rows)

    # Log-fit score
    logfit_rows = []
    for ds in preterm:
        score = subject_logfit_score(ds, seg_list=[1, 2, 3])
        logfit_rows.append(dict(
            id=ds['id'], EP=ds['EP'],
            logfit_score=round(score, 4)))
    df_logfit = pd.DataFrame(logfit_rows)

    # Combined QA
    qa = df_mono.merge(df_logfit, on=['id', 'EP'])
    qa = qa.rename(columns={'mean_violation_pct': 'monotonicity_viol_pct'})

    mono_cutoff = qa['monotonicity_viol_pct'].quantile(0.90)
    logfit_cutoff = qa['logfit_score'].quantile(0.90)

    qa['worst_10pct_mono'] = qa['monotonicity_viol_pct'] > mono_cutoff
    qa['worst_10pct_logfit'] = qa['logfit_score'] > logfit_cutoff
    qa['flag'] = qa['worst_10pct_mono'] | qa['worst_10pct_logfit']

    return df_mono, df_logfit, qa
