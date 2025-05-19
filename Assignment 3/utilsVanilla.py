import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

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


def heston_surface_iv(theta, S0, r, Klist, Tlist):
    """Compute model IV surface for given params theta=[ro, V0, t, k, s]"""
    ro, V0, t, k, s = theta
    iv_model = np.empty((len(Klist), len(Tlist)))
    for i, K in enumerate(Klist):
        for j, T in enumerate(Tlist):
            try:
                price = heston_analytic(S0, V0, r, k, t, s, ro, K, T)
                iv = impvol(price, S0, K, T, r)
                iv_model[i, j] = iv
            except Exception:
                iv_model[i, j] = np.nan
    return iv_model

def calib_obj(theta, Klist, Tlist, ivol_market, S0, r):
    iv_model = heston_surface_iv(theta, S0, r, Klist, Tlist)
    mask = ~np.isnan(ivol_market) & ~np.isnan(iv_model)
    return np.nanmean((ivol_market[mask] - iv_model[mask])**2)

def log_moneyness_axis(Klist, S0):
    """log-moneyness k = ln(K/S0)"""
    return np.log(Klist / S0)

def rel_moneyness_axis(Klist, S0):
    """relative moneyness 𝑚̃ = K/S0 − 1"""
    return Klist / S0 - 1.0

def plot_iv_heatmap_custom(x_axis, Tlist, Z, xlab, title, xlim=(-1, 1)):
    X, Y = np.meshgrid(x_axis, Tlist, indexing='ij')

    plt.figure(figsize=(8, 4.8))
    pc = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(pc, label='Implied Volatility')
    plt.xlabel(xlab)
    plt.ylabel('Maturity $T$ (yrs)')
    plt.title(title)
    plt.xlim(*xlim)
    plt.tight_layout()
    plt.show()

def plot_surface(Klist, Tlist, Z, title):
    from mpl_toolkits.mplot3d import Axes3D
    K, T = np.meshgrid(Klist, Tlist, indexing='ij')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(K, T, Z, rstride=1, cstride=1)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)
    plt.show()

def error_metrics(ivol_market, ivol_model):
    mask = ~np.isnan(ivol_market) & ~np.isnan(ivol_model)
    err = ivol_model[mask] - ivol_market[mask]
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    maxerr = np.max(np.abs(err))
    return rmse, mae, maxerr