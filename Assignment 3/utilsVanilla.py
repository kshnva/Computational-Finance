import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import brentq
import os
from functools import partial
from scipy.optimize import least_squares

def heston_mc(S0, V0, r, k, t, s, ro, K, T, M, dt, seed=42):
    rng = np.random.default_rng(seed)
    N = int(T / dt)
    S = np.full(M, S0)
    V = np.full(M, V0)
    Z1 = rng.standard_normal((M, N))
    Z2 = rng.standard_normal((M, N))
    dW2 = np.sqrt(dt) * Z1
    dW1 = np.sqrt(dt) * (ro * Z1 + np.sqrt(1 - ro**2) * Z2)
    for n in range(N):
        Vp = np.maximum(V, 0.0)
        V = V + k * (t - Vp) * dt + s * np.sqrt(Vp) * dW2[:, n]
        V = np.maximum(V, 0.0)
        S = S * np.exp((r - 0.5 * Vp) * dt + np.sqrt(Vp) * dW1[:, n])
    dpay = np.exp(-r * T) * np.maximum(S - K, 0.0)
    m = dpay.mean()
    h = 1.96 * dpay.std(ddof=1) / np.sqrt(M)
    return m, m-h, m+h

def heston_cf(u, S0, V0, r, T, k, t, s, ro):
    i = 1j
    x = k - s * ro * i * u
    d = np.sqrt(x**2 + s**2 * (u**2 + i*u))
    g = (x - d) / (x + d)
    C = k * t / s**2 * ((x - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = V0 / s**2 * (x - d) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    return np.exp(i * u * (np.log(S0) + r * T) + C + D)

def heston_int(u, S0, K, T, r, V0, k, t, s, ro, j):
    i = 1j
    F = S0 * np.exp(r * T)
    u_ = u - i if j == 1 else u
    cf = heston_cf(u_, S0, V0, r, T, k, t, s, ro)
    if j == 1:
        return np.real(np.exp(-i * u * np.log(K)) * cf / (i * u * F))
    else:
        return np.real(np.exp(-i * u * np.log(K)) * cf / (i * u))


def heston_analytic(S0, V0, r, k, t, s, ro, K, T):
    P1 = 0.5 + (1/np.pi) * quad(lambda u: heston_int(u, S0, K, T, r, V0, k, t, s, ro, 1), 1e-8, 50, limit=100)[0]
    P2 = 0.5 + (1/np.pi) * quad(lambda u: heston_int(u, S0, K, T, r, V0, k, t, s, ro, 2), 1e-8, 50, limit=100)[0]
    return S0 * P1 - np.exp(-r * T) * K * P2


def grid(S0, V0, r, k, t, s, ro, Klist, Tlist, M, dt, seed=42):
    out = {}
    for K in Klist:
        for T in Tlist:
            mc, lo, hi = heston_mc(S0, V0, r, k, t, s, ro, K, T, M, dt, seed)
            ana = heston_analytic(S0, V0, r, k, t, s, ro, K, T)
            out[(K, T)] = {'MC': (mc, lo, hi), 'AN': ana}
    return out

def bs_call(S, K, T, r, sig):
    if T == 0: return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def impvol(C, S, K, T, r):
    if C < max(S - K * np.exp(-r * T), 0) + 1e-8:
        return np.nan
    def obj(sig): return bs_call(S, K, T, r, sig) - C
    try:
        return brentq(obj, 1e-6, 5.0, maxiter=100)
    except Exception:
        return np.nan

def visualize_surface(Klist, Tlist, Z, title, zlab):
    K, T = np.meshgrid(Klist, Tlist, indexing='ij')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K, T, Z, cmap='viridis')
    ax.set_xlabel('Strike K')
    ax.set_ylabel('Maturity T')
    ax.set_zlabel(zlab)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def build_surfaces(out, S0, r, Klist, Tlist):
    pmc = np.empty((len(Klist), len(Tlist)))
    pan = np.empty((len(Klist), len(Tlist)))
    ivmc = np.empty((len(Klist), len(Tlist)))
    ivan = np.empty((len(Klist), len(Tlist)))
    for i, K in enumerate(Klist):
        for j, T in enumerate(Tlist):
            mc = out[(K, T)]["MC"][0]
            an = out[(K, T)]["AN"]
            pmc[i, j] = mc
            pan[i, j] = an
            ivmc[i, j] = impvol(mc, S0, K, T, r)
            ivan[i, j] = impvol(an, S0, K, T, r)
    return pmc, pan, ivmc, ivan


# =============================================================================

def load_ivol_surfaces(file_path):
    """Return the pickled dict {date: {'tenors','strikes','vols'}}."""
    return np.load(file_path, allow_pickle=True).item()

def bs_matrix(S0, r, strikes, tenors, ivols):
    """Vectorised BS call prices for a grid of strikes × tenors."""
    K = strikes[None, :]            # shape (1, 100)
    T = tenors[:, None]             # shape (N, 1)
    vols = ivols                    # shape (N, 100)
    d1 = (np.log(S0 / K) + (r + 0.5 * vols**2) * T) / (vols * np.sqrt(T))
    d2 = d1 - vols * np.sqrt(T)
    price = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return price    

def heston_iv_surface(theta, S0, r, strikes, tenors):
    rho, V0, theta_bar, kappa, sigma = theta
    N, M = len(tenors), len(strikes)
    ivols = np.empty((N, M))

    # broadcast through loops (tenors outer for less re-work of r,T params)
    for i, T in enumerate(tenors):
        # vectorise strikes in the inner loop
        prices = np.array([heston_analytic(S0, V0, r, kappa,
                                           theta_bar, sigma, rho,
                                           K, T) for K in strikes])
        # invert to implied vols (scalar Brent per K; could be sped up w/ JAX)
        ivols[i, :] = [impvol(C, S0, K, T, r) for C, K in zip(prices, strikes)]
    return ivols   


def heston_iv_surface_mc(theta, S0, r, strikes, tenors,
                         M=40_000, dt=1/365, seed=42):
    """
    Build an (N, M) matrix of implied vols with *Monte-Carlo* pricing.

    Parameters
    ----------
    theta   : list/np.array  [rho, V0, theta_bar, kappa, sigma]
    S0, r   : spot and risk-free rate
    strikes : (M,) 1-D array of strikes  (same fixed grid as the interp file)
    tenors  : (N,) 1-D array of times to maturity (in years)
    M, dt   : MC control — number of paths and time step
    seed    : RNG seed for reproducibility

    Returns
    -------
    ivols_mc : (N, M) ndarray of implied vols  (NaN where inversion fails)
    """
    rho, V0, theta_bar, kappa, sigma = theta
    ivols_mc = np.empty((len(tenors), len(strikes)))

    # outer loop on maturities keeps RNG calls grouped per maturity
    for i, T in enumerate(tenors):
        for j, K in enumerate(strikes):
            price_mc, *_ = heston_mc(S0, V0, r,
                                     kappa, theta_bar, sigma, rho,
                                     K, T, M, dt, seed+37*i+j)
            ivols_mc[i, j] = impvol(price_mc, S0, K, T, r)

    return ivols_mc

'''
maybe later change to
def calib_objective(theta, market_iv, S0, r, strikes, tenors, iv_surface_func, w=None):
    model_iv = iv_surface_func(theta, S0, r, strikes, tenors)
'''


def calib_objective(theta, market_iv, S0, r, strikes, tenors, w=None):
    """Flattened residual vector for least_squares."""
    model_iv = heston_iv_surface(theta, S0, r, strikes, tenors)
    if w is None:                   # equal weights
        resid = (model_iv - market_iv).ravel()
    else:
        resid = (w * (model_iv - market_iv)).ravel()
    # scrub NaNs that may arise from failed vol inversion
    return resid[~np.isnan(resid)]



def calib_objective(theta, market_iv, S0, r, strikes, tenors, w=None, penalty=1.0):
    """
    Flattened residual vector for least_squares.
    Any NaN/Inf in either market or model IVs is replaced
    by a constant penalty so the vector length never changes.
    """
    model_iv = heston_iv_surface(theta, S0, r, strikes, tenors)
    diff = model_iv - market_iv         

    if w is not None:
        diff *= w
    # replace invalid numbers with a constant penalty 
    bad = ~np.isfinite(diff)
    diff[bad] = penalty

    return diff.ravel()       

def calibrate_heston_ivols(market_iv, S0, r, strikes, tenors, guess=None, bounds=None, weights=None):
    """
    Least-squares calibration on implied vols.
    Returns (theta_hat, optimisation_result).
    """
    if guess is None:
        guess = np.array([-0.4,   # rho
                          0.04,   # V0
                          0.04,   # theta
                          1.5,    # kappa
                          0.3])   # sigma
    if bounds is None:
        lb = [-0.999,  1e-6, 1e-6,  1e-3, 1e-3]
        ub = [ 0.999,  2.0,  2.0,  10.0, 5.0]
        bounds = (lb, ub)

    result = least_squares(calib_objective, guess,
                           args=(market_iv, S0, r, strikes, tenors, weights),
                           bounds=bounds, method='trf',
                           ftol=1e-8, xtol=1e-8, gtol=1e-4,
                           max_nfev=10_000)

    return result.x, result

def error_metrics(model_iv, market_iv):
    """Return RMSE, MAE, MAX on vols (ignoring NaNs)."""
    diff = model_iv - market_iv
    diff = diff[~np.isnan(diff)]
    rmse = np.sqrt(np.mean(diff**2))
    mae  = np.mean(np.abs(diff))
    mxe  = np.max(np.abs(diff))
    return rmse, mae, mxe

def diagnostics_plot(date, strikes, tenors, market_iv, model_iv):
    visualize_surface(strikes, tenors, market_iv.T,
                      f"Market IV surface – {date}", "Implied vol")
    visualize_surface(strikes, tenors, model_iv.T,
                      f"Heston IV surface – {date}", "Implied vol")

def run_calibration_on_date(date, surfaces, S0, r, do_plot=True):
    data = surfaces[date]
    tenors   = data["tenors"]       # (N,)
    strikes  = data["strikes"]      # (100,)
    market_iv= data["vols"]         # (N, 100)

    theta_hat, res = calibrate_heston_ivols(market_iv, S0, r,
                                            strikes, tenors)

    model_iv = heston_iv_surface(theta_hat, S0, r, strikes, tenors)
    rmse, mae, mxe = error_metrics(model_iv, market_iv)

    if do_plot:
        diagnostics_plot(date, strikes, tenors, market_iv, model_iv)

    print(f"=== {date} ===")
    print("theta_hat = [rho, V0, theta, kappa, sigma] :")
    print(theta_hat)
    print(f"RMSE = {rmse:.4e},   MAE = {mae:.4e},   MaxErr = {mxe:.4e}\n")
    return theta_hat, (rmse, mae, mxe), res





"""
Heston-model calibration to S&P 500 implied-volatility surfaces.
----------------------------------------------------------------
*Revised to reuse the user‑supplied pricing routine verbatim whenever it is
importable; otherwise, it falls back to an identical in‑script copy.*

See original header for details of workflow and CLI options.


from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D plots

# =============================================================================
#  BLACK–SCHOLES HELPERS  (unchanged – lifted straight from the user code)   ===
# =============================================================================

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T == 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                 *, tol: float = 1e-6) -> float:
    if price < max(S - K * np.exp(-r * T), 0) + 1e-8:
        return np.nan

    def obj(sig):
        return bs_call(S, K, T, r, sig) - price

    try:
        return brentq(obj, 1e-6, 5.0, xtol=tol, rtol=1e-10, maxiter=100)
    except ValueError:
        return np.nan


# =============================================================================
#  HESTON ANALYTICAL PRICER  – *user code reused as the primary engine*      ===
# =============================================================================

try:
    # If the user already imported their module (e.g. notebook cell), grab it:
    heston_analytic  # noqa: F821 – NameError if not defined

    def heston_price(S0: float, V0: float, r: float, kappa: float, theta: float,
                     sigma: float, rho: float, K: float, T: float) -> float:  # noqa: D401
        return heston_analytic(S0, V0, r, kappa, theta, sigma, rho, K, T)  # type: ignore[name-defined]

except NameError:
    # ------------------------------------------------------------------
    #  Stand‑alone fallback – the *exact same* formulas found in the user
    #  snippet (copy‑pasted here to honour the reuse requirement).
    # ------------------------------------------------------------------

    def _heston_cf(u, S0, V0, r, T, kappa, theta, sigma, rho):
        i = 1j
        x = kappa - sigma * rho * i * u
        d = np.sqrt(x**2 + sigma**2 * (u**2 + i * u))
        g = (x - d) / (x + d)
        C = kappa * theta / sigma**2 * (
            (x - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        D = V0 / sigma**2 * (x - d) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        return np.exp(i * u * (np.log(S0) + r * T) + C + D)

    def _p_int(u, S0, K, T, r, V0, kappa, theta, sigma, rho, j):
        i = 1j
        if j == 1:
            cf = _heston_cf(u - i, S0, V0, r, T, kappa, theta, sigma, rho)
            return np.real(np.exp(-i * u * np.log(K)) * cf / (i * u * S0 * np.exp(r * T)))
        else:
            cf = _heston_cf(u, S0, V0, r, T, kappa, theta, sigma, rho)
            return np.real(np.exp(-i * u * np.log(K)) * cf / (i * u))

    def heston_price(S0: float, V0: float, r: float, kappa: float, theta: float,
                     sigma: float, rho: float, K: float, T: float,
                     *, _u_max: float = 50.0) -> float:
        P1 = 0.5 + (1 / np.pi) * quad(_p_int, 1e-8, _u_max,
                                      args=(S0, K, T, r, V0, kappa, theta, sigma, rho, 1),
                                      limit=150, epsabs=1e-8, epsrel=1e-8)[0]
        P2 = 0.5 + (1 / np.pi) * quad(_p_int, 1e-8, _u_max,
                                      args=(S0, K, T, r, V0, kappa, theta, sigma, rho, 2),
                                      limit=150, epsabs=1e-8, epsrel=1e-8)[0]
        return S0 * P1 - np.exp(-r * T) * K * P2

# Vectorised helper (strike, tenor arrays) – excludes fixed params
_vec_heston_price = np.vectorize(heston_price, excluded={0, 1, 2, 3, 4, 5, 6})


# =============================================================================
#  CALIBRATION ROUTINES                                                      ===
# =============================================================================

def _model_iv(params: np.ndarray, S0: float, r: float,
              strikes: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    rho, V0, theta, kappa, sigma = params
    KK, TT = np.meshgrid(strikes, tenors, indexing="ij")
    prices = np.empty_like(KK)
    for i, K in enumerate(strikes):
        for j, T in enumerate(tenors):
            prices[i, j] = heston_price(S0, V0, r, kappa, theta, sigma, rho, K, T)
    iv = np.vectorize(implied_vol)(prices, S0, KK, TT, r)
    return iv


def _residuals(params, market_iv, S0, r, strikes, tenors, mask):
    pred = _model_iv(params, S0, r, strikes, tenors)
    diff = pred - market_iv
    return np.nan_to_num(diff[mask]).ravel()


def calibrate(market_iv, S0, r, strikes, tenors, mask,
              initial=(-0.5, 0.04, 0.04, 1.5, 0.3)):
    bounds = ([-0.999, 1e-6, 1e-6, 1e-3, 1e-3], [0.999, 2.0, 2.0, 15.0, 5.0])
    res = least_squares(_residuals, initial, bounds=bounds,
                        args=(market_iv, S0, r, strikes, tenors, mask),
                        ftol=1e-6, xtol=1e-6, max_nfev=300, method="trf")
    return res


# =============================================================================
#  PLOT HELPERS                                                             ===
# =============================================================================

def plot_surface(strikes, tenors, iv, title):
    K_mesh, T_mesh = np.meshgrid(strikes, tenors, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(K_mesh, T_mesh, iv, rstride=1, cstride=1, linewidth=0.5)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (y)")
    ax.set_zlabel("Implied vol")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  MAIN DRIVER                                                              ===
# =============================================================================

def run_date(date, data, *, S0=None, r=0.0, nth=2):
    rec = data[date]
    tenors = rec["tenors"].astype(float)
    strikes = rec["strikes"].astype(float)
    market_iv = rec["vols"].T.astype(float)  # (K,T)

    if S0 is None:
        idx = np.argmin(np.var(market_iv, axis=1))
        S0 = strikes[idx]
        warnings.warn(f"S0 not supplied – using heuristic ATM strike: {S0:.2f}")

    # sub‑sample strikes for speed
    mask_strike = np.arange(0, len(strikes), nth)
    calib_iv = market_iv[mask_strike, :]
    calib_strikes = strikes[mask_strike]
    mask = ~np.isnan(calib_iv)

    res = calibrate(calib_iv, S0, r, calib_strikes, tenors, mask)

    model_full = _model_iv(res.x, S0, r, strikes, tenors)
    diff = model_full - market_iv
    rmse = np.sqrt(np.nanmean(diff**2))
    mae = np.nanmean(np.abs(diff))
    max_err = np.nanmax(np.abs(diff))

    print(f"\n{date}  – fitted params (ρ, V0, θ, κ, σ):", res.x)
    print(f"Errors:  RMSE={rmse:.4f}   MAE={mae:.4f}   MAX={max_err:.4f}\n")

    plot_surface(strikes, tenors, market_iv, f"{date} – Market IV")
    plot_surface(strikes, tenors, model_full, f"{date} – Heston IV (fit)")

    return {
        "date": date,
        "params": res.x,
        "rmse": rmse,
        "mae": mae,
        "max": max_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", default="/data/interp_ivol_surfaces.npy")
    ap.add_argument("--dates", nargs="*", help="Dates yyyy-mm-dd … (default first two)")
    ap.add_argument("--S0", type=float)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--nth", type=int, default=2, help="Use every n‑th strike (speed)")
    args = ap.parse_args()

    data = np.load(Path(args.datafile), allow_pickle=True).item()
    all_dates = sorted(data.keys())
    dates = args.dates if args.dates else all_dates[:2]
    print("Running on:", ", ".join(dates))

    results = [run_date(d, data, S0=args.S0, r=args.r, nth=args.nth) for d in dates]

    print("\nSummary:")
    for res in results:
        print(f"{res['date']}  – RMSE {res['rmse']:.4f}  params {res['params']}")


if __name__ == "__main__":
    main()
"""