import numpy as np
from scipy.integrate import quad

def heston_mc_euler(S0, V0, r, kappa, theta, sigma, rho, K, T, M=10000, dt=1/252, seed=42):
    rng = np.random.default_rng(seed)
    N = int(T / dt)
    S = np.full(M, S0)
    V = np.full(M, V0)
    Z1 = rng.standard_normal((M, N))
    Z2 = rng.standard_normal((M, N))
    dW2 = np.sqrt(dt) * Z1
    dW1 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
    for n in range(N):
        Vpos = np.maximum(V, 0.0)
        V = V + kappa * (theta - Vpos) * dt + sigma * np.sqrt(Vpos) * dW2[:, n]
        V = np.maximum(V, 0.0)  # Full truncation
        S = S * np.exp((r - 0.5 * Vpos) * dt + np.sqrt(Vpos) * dW1[:, n])
    discounted_payoff = np.exp(-r * T) * np.maximum(S - K, 0.0)
    mean = discounted_payoff.mean()
    h = 1.96 * discounted_payoff.std(ddof=1) / np.sqrt(M)
    return mean, mean-h, mean+h


# Semi-closed form for Heston model
def heston_characteristic_func(u, S0, V0, r, T, kappa, theta, sigma, rho):
    i = 1j
    xi = kappa - sigma * rho * i * u
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + i*u))
    g = (xi - d) / (xi + d)
    C = kappa * theta / sigma**2 * ((xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = V0 / sigma**2 * (xi - d) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    return np.exp(i * u * (np.log(S0) + r * T) + C + D)

def heston_integrand(u, S0, K, T, r, V0, kappa, theta, sigma, rho, j):
    i = 1j
    if j == 1:
        u_shift = u - i
    else:
        u_shift = u
    cf = heston_characteristic_func(u_shift, S0, V0, r, T, kappa, theta, sigma, rho)
    return np.real(np.exp(-i * u * np.log(K)) * cf / (i * u))

def heston_semi_closed(S0, V0, r, kappa, theta, sigma, rho, K, T):
    P1 = 0.5 + (1/np.pi) * quad(lambda u: heston_integrand(u, S0, K, T, r, V0, kappa, theta, sigma, rho, 1), 1e-8, 100, limit=100)[0]
    P2 = 0.5 + (1/np.pi) * quad(lambda u: heston_integrand(u, S0, K, T, r, V0, kappa, theta, sigma, rho, 2), 1e-8, 100, limit=100)[0]
    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price

def price_grid(S0, V0, r, kappa, theta, sigma, rho, K_list, T_list, M=10000, dt=1/252, seed=42):
    results = {}
    for K in K_list:
        for T in T_list:
            mc_price, mc_lo, mc_hi = heston_mc_euler(S0, V0, r, kappa, theta, sigma, rho, K, T, M, dt, seed)
            heston_price = heston_semi_closed(S0, V0, r, kappa, theta, sigma, rho, K, T)
            results[(K, T)] = {
                "MC": (mc_price, mc_lo, mc_hi),
                "HestonSemiClosed": heston_price
            }
    return results
