"""
T2 relaxometry models and fit utilities for CMBI coursework.

Single source of truth for Tasks 2–7. Exposes:
  - Signal models: mono_exp, bi_exp_fixed, bi_exp_free
  - One-compartment fitters: fit_two_point, fit_log_linear, fit_wls, fit_nlls
  - Two-compartment fitters: fit_bi_fixed, fit_bi_free
  - Uncertainty: parametric_bootstrap_mono, parametric_bootstrap_bi_fixed
  - Model comparison: aicc
  - MCMC: mcmc_bi_fixed
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# Signal models
# ----------------------------------------------------------------------
def mono_exp(TE, S0, T2):
    '''Mono-exponential: S(TE) = S0 * exp(-TE / T2).'''
    return S0 * np.exp(-np.asarray(TE, float) / T2)


def bi_exp_fixed(TE, S0, v, T2s, T2l):
    '''Two-parameter bi-exp with T2s, T2l fixed by caller.'''
    TE = np.asarray(TE, float)
    return S0 * (v * np.exp(-TE / T2s) + (1.0 - v) * np.exp(-TE / T2l))


def bi_exp_free(TE, S0, v, T2s, T2l):
    '''Four-parameter bi-exp (all params free).'''
    return bi_exp_fixed(TE, S0, v, T2s, T2l)


# ----------------------------------------------------------------------
# One-compartment fitters
# ----------------------------------------------------------------------
def fit_two_point(TE_a, S_a, TE_b, S_b):
    '''Analytical T2 from two echoes.'''
    if S_a <= 0 or S_b <= 0 or S_a == S_b:
        return np.nan, np.nan
    T2 = (TE_b - TE_a) / np.log(S_a / S_b)
    if T2 <= 0 or not np.isfinite(T2):
        return np.nan, np.nan
    return float(S_a * np.exp(TE_a / T2)), float(T2)


def fit_log_linear(TE, signal):
    '''Log-linear OLS. Requires S > 0 for every echo.'''
    signal = np.asarray(signal, float)
    if np.any(signal <= 0):
        return np.nan, np.nan
    A = np.column_stack([np.ones_like(TE), TE])
    p, *_ = np.linalg.lstsq(A, np.log(signal), rcond=None)
    if p[1] >= 0:
        return np.nan, np.nan
    return float(np.exp(p[0])), float(-1.0 / p[1])


def fit_wls(TE, signal):
    '''Weighted log-linear LS with w_k = S_k^2.'''
    signal = np.asarray(signal, float)
    if np.any(signal <= 0):
        return np.nan, np.nan
    A = np.column_stack([np.ones_like(TE), TE])
    W = np.diag(signal ** 2)
    try:
        p = np.linalg.solve(A.T @ W @ A, A.T @ W @ np.log(signal))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    if p[1] >= 0:
        return np.nan, np.nan
    return float(np.exp(p[0])), float(-1.0 / p[1])


def fit_nlls(TE, signal):
    '''NLLS (TRF) warm-started from log-linear. Bounds: S0 >= 0, 5 <= T2 <= 3000 ms.'''
    signal = np.asarray(signal, float)
    S0_init, T2_init = fit_log_linear(TE, signal)
    if not np.isfinite(T2_init):
        S0_init, T2_init = float(signal[0]), 80.0
    try:
        popt, _ = curve_fit(mono_exp, TE, signal,
                            p0=[S0_init, T2_init],
                            bounds=([0, 5], [np.inf, 3000]),
                            method='trf', maxfev=5000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


# ----------------------------------------------------------------------
# Two-compartment fitters
# ----------------------------------------------------------------------
def fit_bi_fixed(TE, signal, T2s, T2l, v_init=0.10):
    '''2-param NLLS with T2 values fixed. Returns [S0, v].'''
    signal = np.asarray(signal, float)
    model  = lambda TE, S0, v: bi_exp_fixed(TE, S0, v, T2s, T2l)
    try:
        popt, _ = curve_fit(model, TE, signal,
                            p0=[float(signal[0]), v_init],
                            bounds=([0, 0], [np.inf, 1]),
                            method='trf', maxfev=5000)
        return list(popt)
    except Exception:
        return [np.nan, np.nan]


def fit_bi_free(TE, signal, v_init=0.10, T2s_init=20.0, T2l_init=80.0):
    '''4-param NLLS with non-overlapping T2 ranges for ordering.
    T2_short in [5, 50] ms; T2_long in [50, 3000] ms.'''
    signal = np.asarray(signal, float)
    try:
        popt, _ = curve_fit(bi_exp_free, TE, signal,
                            p0=[float(signal[0]), v_init, T2s_init, T2l_init],
                            bounds=([0,      0, 5,   50  ],
                                    [np.inf, 1, 50, 3000]),
                            method='trf', maxfev=10000)
        return list(popt)
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]


# ----------------------------------------------------------------------
# Uncertainty - parametric bootstrap
# ----------------------------------------------------------------------
def parametric_bootstrap_mono(TE, signal, n_boot=200, seed=0):
    '''Bootstrap for mono-exp NLLS. Returns (T2_hat, CI_low, CI_high) for T2.'''
    TE = np.asarray(TE, float); signal = np.asarray(signal, float)
    S0h, T2h = fit_nlls(TE, signal)
    if not np.isfinite(T2h):
        return np.nan, np.nan, np.nan
    pred  = mono_exp(TE, S0h, T2h)
    sigma = float(np.std(signal - pred, ddof=1))
    rng   = np.random.default_rng(seed)
    T2s   = np.empty(n_boot)
    for b in range(n_boot):
        S_b = pred + rng.normal(0, sigma, size=TE.shape)
        _, T2_b = fit_nlls(TE, S_b)
        T2s[b] = T2_b
    T2s = T2s[np.isfinite(T2s)]
    if len(T2s) < 10:
        return T2h, np.nan, np.nan
    lo, hi = np.percentile(T2s, [2.5, 97.5])
    return T2h, float(lo), float(hi)


def parametric_bootstrap_bi_fixed(TE, signal, T2s, T2l, n_boot=200, seed=0):
    '''Bootstrap for bi-exp fixed. Returns (v_hat, CI_low, CI_high) for v.'''
    TE = np.asarray(TE, float); signal = np.asarray(signal, float)
    S0h, vh = fit_bi_fixed(TE, signal, T2s, T2l, v_init=0.10)
    if not np.isfinite(vh):
        return np.nan, np.nan, np.nan
    pred  = bi_exp_fixed(TE, S0h, vh, T2s, T2l)
    sigma = float(np.std(signal - pred, ddof=1))
    rng   = np.random.default_rng(seed)
    v_arr = np.empty(n_boot)
    for b in range(n_boot):
        S_b = pred + rng.normal(0, sigma, size=TE.shape)
        _, v_b = fit_bi_fixed(TE, S_b, T2s, T2l, v_init=vh)
        v_arr[b] = v_b
    v_arr = v_arr[np.isfinite(v_arr)]
    if len(v_arr) < 10:
        return vh, np.nan, np.nan
    lo, hi = np.percentile(v_arr, [2.5, 97.5])
    return vh, float(lo), float(hi)


# ----------------------------------------------------------------------
# Model comparison - AICc (Lecture 06)
# ----------------------------------------------------------------------
def aicc(residuals, n_params):
    '''AICc for Gaussian residuals. n_params must include sigma (so +1 vs
    the number of fit parameters). AICc = 2N + K*log(SSR/K) + 2N(N+1)/(K-N-1).
    Smaller is better. Use AICc (not AIC) when K/N < 40 (Lecture 06).'''
    r = np.asarray(residuals, float)
    K = len(r)
    N = n_params
    if K - N - 1 <= 0:
        return np.nan
    SSR = float(np.sum(r ** 2))
    return 2*N + K*np.log(SSR/K) + 2*N*(N+1) / (K - N - 1)


def akaike_weights(aicc_values):
    '''Akaike weights: relative probability of each model being the best.'''
    a = np.asarray(aicc_values, float)
    d = a - np.nanmin(a)
    w = np.exp(-d / 2)
    return w / np.nansum(w)


# ----------------------------------------------------------------------
# MCMC for v (Metropolis-Hastings, Dingwall 2016 style with Beta prior)
# ----------------------------------------------------------------------
def mcmc_bi_fixed(TE, signal, T2s, T2l,
                  n_iter=50000, burn_in=10000, thin=10, step=0.02,
                  prior_alpha=2.0, prior_beta=10.0, seed=0,
                  return_diagnostics=False):
    '''Metropolis-Hastings MCMC for v in the fixed-T2 bi-exp model.

    S0 is analytically profiled out at each proposal, given v and fixed
    T2 values. Prior on v is Beta(alpha, beta), which is a simple
    two-component analogue of a Dirichlet prior on compartment fractions.
    Noise sigma is estimated once from the initial fixed-T2 fit and then
    held fixed throughout the chain.

    Returns
    post : ndarray
        Posterior samples of v after burn-in and thinning.
    v_hat : float
        Posterior median of v.
    ci : tuple
        95% posterior interval for v.
    diag : dict, optional
        Diagnostics, only returned if return_diagnostics=True.
    '''
    TE = np.asarray(TE, float)
    signal = np.asarray(signal, float)

    # Initial fixed-T2 fit to estimate sigma and provide chain start
    S0h, vh = fit_bi_fixed(TE, signal, T2s, T2l, v_init=0.10)
    if not np.isfinite(vh):
        if return_diagnostics:
            return np.array([]), np.nan, (np.nan, np.nan), {}
        return np.array([]), np.nan, (np.nan, np.nan)

    pred = bi_exp_fixed(TE, S0h, vh, T2s, T2l)
    sigma = float(np.std(signal - pred, ddof=1))
    sigma = max(sigma, 1e-8)

    def log_posterior(v):
        # Prior support
        if v <= 0 or v >= 1:
            return -np.inf

        # Beta(alpha, beta) prior, up to an additive constant
        log_prior = (prior_alpha - 1.0) * np.log(v) + (prior_beta - 1.0) * np.log(1.0 - v)

        # Profile S0 analytically by least squares, given v
        basis = v * np.exp(-TE / T2s) + (1.0 - v) * np.exp(-TE / T2l)
        denom = np.dot(basis, basis)
        if denom <= 0:
            return -np.inf

        S0_profile = np.dot(basis, signal) / denom
        if S0_profile <= 0 or not np.isfinite(S0_profile):
            return -np.inf

        resid = signal - S0_profile * basis
        log_like = -0.5 * np.sum((resid / sigma) ** 2)

        return float(log_prior + log_like)

    rng = np.random.default_rng(seed)

    v_curr = float(vh)
    lp_curr = log_posterior(v_curr)

    samples = np.empty(n_iter, dtype=float)
    accepted = 0

    for i in range(n_iter):
        v_prop = v_curr + rng.normal(0.0, step)
        lp_prop = log_posterior(v_prop)

        if np.log(rng.uniform()) < (lp_prop - lp_curr):
            v_curr = v_prop
            lp_curr = lp_prop
            accepted += 1

        samples[i] = v_curr

    # Burn-in + thinning
    post = samples[burn_in::thin]

    if len(post) == 0:
        if return_diagnostics:
            return np.array([]), np.nan, (np.nan, np.nan), {
                'acceptance_rate': accepted / n_iter,
                'sigma_fixed': sigma,
                'v_start': vh,
                'n_iter': n_iter,
                'burn_in': burn_in,
                'thin': thin
            }
        return np.array([]), np.nan, (np.nan, np.nan)

    v_hat = float(np.median(post))
    lo, hi = np.percentile(post, [2.5, 97.5])

    diag = {
        'acceptance_rate': accepted / n_iter,
        'sigma_fixed': sigma,
        'v_start': vh,
        'n_iter': n_iter,
        'burn_in': burn_in,
        'thin': thin,
        'n_kept': len(post)
    }

    if return_diagnostics:
        return post, v_hat, (float(lo), float(hi)), diag
    return post, v_hat, (float(lo), float(hi))


# ============================================================================
# Task 5: multi-compartment NNLS and three-compartment NLLS
# ============================================================================
from scipy.optimize import nnls

# Log-spaced grid: 101 points from 10 to 2000 ms 
T2_GRID = np.logspace(np.log10(10), np.log10(2000), 101)


def make_kernel(TE, T2_grid=T2_GRID):
    '''NNLS kernel K_{ij} = exp(-TE_i / T2_j).'''
    TE = np.asarray(TE, float)
    return np.exp(-TE[:, None] / T2_grid[None, :])


def nnls_plain(TE, signal, T2_grid=T2_GRID):
    '''Unregularised NNLS. Returns (amplitudes, misfit).'''
    K = make_kernel(TE, T2_grid)
    a, rnorm = nnls(K, np.asarray(signal, float), maxiter=5000)
    return a, float(rnorm)


def nnls_regularised(TE, signal, mu, T2_grid=T2_GRID):
    '''Regularised NNLS via augmented system [K; mu*I][a] = [S; 0].
    Implements Tikhonov penalty mu^2 ||a||^2 inside NNLS (Whittall 1997).'''
    K = make_kernel(TE, T2_grid)
    J = K.shape[1]
    K_aug = np.vstack([K, mu * np.eye(J)])
    s_aug = np.concatenate([np.asarray(signal, float), np.zeros(J)])
    a, _ = nnls(K_aug, s_aug, maxiter=5000)
    misfit = float(np.linalg.norm(np.asarray(signal, float) - K @ a))
    return a, misfit


def pick_mu_chi2(TE, signal, tol=1.02, mu_grid=None, T2_grid=T2_GRID):
    '''Whittall chi^2 criterion: smallest mu such that regularised misfit
    does not exceed tol * plain misfit. Default tol=1.02 (2%).'''
    if mu_grid is None:
        mu_grid = np.logspace(-3, 2, 40)
    _, misfit_plain = nnls_plain(TE, signal, T2_grid)
    target = tol * misfit_plain
    best_mu, best_a, best_mf = None, None, None
    for mu in mu_grid:
        a, mf = nnls_regularised(TE, signal, mu, T2_grid)
        if mf <= target:
            best_mu, best_a, best_mf = mu, a, mf
        else:
            break
    if best_mu is None:
        best_mu = mu_grid[0]
        best_a, best_mf = nnls_regularised(TE, signal, best_mu, T2_grid)
    return best_mu, best_a, misfit_plain, best_mf


def mwf_from_spectrum(a, T2_grid=T2_GRID, band=(10.0, 40.0)):
    '''Myelin water fraction from NNLS spectrum: area in T2 band (Mädler 2008).'''
    tot = a.sum()
    if tot <= 0:
        return np.nan
    mask = (T2_grid >= band[0]) & (T2_grid <= band[1])
    return float(a[mask].sum() / tot)


def three_exp_fixed(TE, S0, v1, v2, T2_short=20.0, T2_mid=80.0, T2_long=2000.0):
    '''Dingwall 2016 3-compartment model. v3 = 1 - v1 - v2 (simplex).'''
    TE = np.asarray(TE, float)
    v3 = 1.0 - v1 - v2
    return S0 * (v1 * np.exp(-TE / T2_short)
                 + v2 * np.exp(-TE / T2_mid)
                 + v3 * np.exp(-TE / T2_long))


def fit_three_fixed(TE, signal, T2_short=20.0, T2_mid=80.0, T2_long=2000.0):
    '''NLLS for 3-compartment. Simplex v1+v2+v3=1 enforced
    via reparametrisation: fit (v1, alpha) in [0,1]^2 with
        v1 = fit_param_1
        v2 = (1 - v1) * alpha         so v2 in [0, 1-v1]
        v3 = (1 - v1) * (1 - alpha)   so v3 in [0, 1-v1]
    This guarantees v1, v2, v3 >= 0 and sum = 1 for any (v1, alpha) in [0,1]^2.
    '''
    signal = np.asarray(signal, float)

    def model(TE, S0, v1, alpha):
        v2 = (1.0 - v1) * alpha
        v3 = (1.0 - v1) * (1.0 - alpha)
        TEa = np.asarray(TE, float)
        return S0 * (v1 * np.exp(-TEa / T2_short)
                     + v2 * np.exp(-TEa / T2_mid)
                     + v3 * np.exp(-TEa / T2_long))

    try:
        popt, _ = curve_fit(model, TE, signal,
                            p0=[float(signal[0]), 0.10, 0.95],
                            bounds=([0, 0, 0], [np.inf, 1, 1]),
                            method='trf', maxfev=10000)
        S0, v1, alpha = popt
        v2 = (1.0 - v1) * alpha
        return [S0, v1, v2]
    except Exception:
        return [np.nan] * 3
    

# ============================================================================
# Task 6: MAP prior fit (Bayesian fixed-T2 bi-exp with Gaussian penalty on v)
# ============================================================================

def fit_bi_fixed_map(TE, signal, T2s, T2l, mu_v, sigma_prior, sigma_noise):
    '''MAP fit: fixed-T2 bi-exp with Gaussian prior N(mu_v, sigma_prior^2) on v.

    Returns [S0, v] or [NaN, NaN] on failure.
    '''
    TE = np.asarray(TE, float)
    signal = np.asarray(signal, float)

    sigma_prior = max(float(sigma_prior), 1e-9)
    sigma_noise = max(float(sigma_noise), 1e-9)

    def residual(theta):
        S0, v = theta
        r_data = (signal - bi_exp_fixed(TE, S0, v, T2s, T2l)) / sigma_noise
        r_prior = (v - mu_v) / sigma_prior
        return np.concatenate([r_data, [r_prior]])

    try:
        res = least_squares(
            residual,
            x0=[float(signal[0]), mu_v],
            bounds=([0, 0], [np.inf, 1]),
            method='trf',
            max_nfev=5000
        )
        return [float(res.x[0]), float(res.x[1])]
    except Exception:
        return [np.nan, np.nan]