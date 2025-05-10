import numpy as np
from numpy.random import default_rng
from scipy.stats import norm

def heston_mc(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, seed):
    # sets up the number of time steps and paths for the Monte Carlo simulation
    random_stream = np.random.default_rng(seed)
    N = int(T/dt)

    # allocate paths for the stock and variance. We add 1 to the number of time steps to store the initial state
    # euler
    euler_S = np.empty((M, N+1))
    euler_S[:,0] = S0
    euler_V = np.empty((M, N+1))
    euler_V[:,0] = V0

    # milstein
    milstein_S = np.empty_like(euler_S)
    milstein_S[:,0] = S0
    milstein_V = np.empty_like(euler_V)
    milstein_V[:,0] = V0

    # Create correlated random variables
    Z1 = random_stream.standard_normal((M, N))     
    Z2 = random_stream.standard_normal((M, N))          

    dW2 = np.sqrt(dt) * Z1                               #  (variance)
    dW1 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)  # (price)

    for n in range(N):
        # full-truncation: use V⁺_t in √V and drift terms. full truncation is the safer baseline because it introduces the least systematic error.
        Vpos_e = np.maximum(euler_V[:, n], 0.0)
        Vpos_m = np.maximum(milstein_V[:, n], 0.0)

        # Euler scheme
        euler_V[:, n+1] = (euler_V[:, n]
                           + kappa*(theta - Vpos_e)*dt
                           + xi*np.sqrt(Vpos_e)*dW2[:, n])
        euler_V[:, n+1] = np.maximum(euler_V[:, n+1], 0.0)     

        euler_S[:, n+1] = (euler_S[:, n] *
                           np.exp((r - 0.5*Vpos_e)*dt
                                  + np.sqrt(Vpos_e)*dW1[:, n]))

        # Milstein scheme
        milstein_V[:, n+1] = (milstein_V[:, n]
                              + kappa*(theta - Vpos_m)*dt
                              + xi*np.sqrt(Vpos_m)*dW2[:, n]
                              + 0.25*xi**2*dt*(dW2[:, n]**2 - dt))     
        milstein_V[:, n+1] = np.maximum(milstein_V[:, n+1], 0.0)

        milstein_S[:, n+1] = (milstein_S[:, n] *
                              np.exp((r - 0.5*Vpos_m)*dt
                                     + np.sqrt(Vpos_m)*dW1[:, n]
                                     + 0.5 * Vpos_m * dt * (dW1[:, n]**2 / dt - 1)))  
        
    # calculate the average stock price
    # Exclude the initial state (column 0) when calculating the average stock price for Milstein
    avg_m = milstein_S[:, 1:].mean(axis=1)
    avg_e = euler_S[:, 1:].mean(axis=1)      # exclude column 0 for strict average

    # discounted payoff
    payoff_e = np.exp(-r*T) * np.maximum(avg_e - K, 0.0)
    payoff_m = np.exp(-r*T) * np.maximum(avg_m - K, 0.0)

    # 95 % confidence intervals for the option prices
    def ci(x):
        mean = x.mean()
        h = 1.96 * x.std(ddof=1) / np.sqrt(M)
        return mean, mean - h, mean + h

    price_e, lo_e, hi_e = ci(payoff_e)
    price_m, lo_m, hi_m = ci(payoff_m)

    return (price_e, lo_e, hi_e), (price_m, lo_m, hi_m)

def gbm(
    paths: int,
    s0: float,
    strike: float,
    expiry: float,
    rate: float,
    vol: float,
    dt: float,
    seed: int = 0,
):
    steps = int(expiry / dt)
    rng = default_rng(seed)

    s = np.full(paths, s0, dtype=float)
    running_sum = np.zeros(paths, dtype=float)

    drift = (rate - 0.5 * vol * vol) * dt
    diffusion = vol * np.sqrt(dt)

    for _ in range(steps):
        z = rng.standard_normal(paths)
        s *= np.exp(drift + diffusion * z)
        running_sum += s

    average = running_sum / steps
    discounted_payoff = np.exp(-rate * expiry) * np.maximum(average - strike, 0.0)

    price = discounted_payoff.mean()
    half_ci = 1.96 * discounted_payoff.std(ddof=1) / np.sqrt(paths)
    return price, half_ci


def geometric_analytical_asian(S0, K, r, sigma, T, N):
    """
    Discrete sampling Geometric Asian option price using the analytical formula.
    """
    # effective vol of the geometr ic average
    sigma_bar = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1)))

    # degenerate case: no randomness
    if sigma_bar < 1e-16:
        # average = S0, payoff = exp(-rT)*max(S0-K, 0)
        return np.exp(-r * T) * max(S0 - K, 0.0)

    # otherwise do the usual d1/d2
    r_bar  = 0.5 * (r - 0.5 * sigma**2 + sigma_bar**2)
    sqrtT  = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r_bar + 0.5*sigma_bar**2)*T) / (sigma_bar * sqrtT)
    d2 = d1 - sigma_bar * sqrtT

    return np.exp(-r * T) * (S0*np.exp(r_bar*T)*norm.cdf(d1) - K*norm.cdf(d2))



def mc_geometric_asian(paths, S0, K, r, sigma, T, N, seed):

    # Monte Carlo simulation for geometric Asian, compared to analytic

    dt = T / N
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((paths, N))

    logS = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    logS_full = np.hstack((np.zeros((paths,1)), logS))

    # compute discrete geometric average
    G = np.exp(logS_full.mean(axis=1) + np.log(S0))
    payoffs = np.exp(-r * T) * np.maximum(G - K, 0)

    return payoffs.mean(), 1.96 * payoffs.std(ddof=1) / np.sqrt(paths)


def control_variate_MC(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed=None):
    """
    Monte Carlo pricing of an arithmetic Asian call under Heston, using
    a geometric Asian GBM as a control variate (Milstein discretization).

    Returns a dict with:
      'plain'       : [price,   st.error, variance] of arithmetic payoff
      'control_var' : [price,   st.error, variance] after CV correction
      'analy_price' : analytic geometric Asian price (control variate target)
      'payoffs'     : [Y, X] raw discounted payoffs arrays for Heston (Y)
                      and geometric GBM (X)
    """
    N = int(T/dt)
    rng = np.random.default_rng(seed)

    S_h = np.full(M, S0, dtype=float)
    V   = np.full(M, V0, dtype=float)
    S_g = np.full(M, S0, dtype=float)

    sum_S_h   = np.zeros(M, dtype=float)
    sum_log_g = np.zeros(M, dtype=float)


    for _ in range(N):
        sum_S_h   += S_h
        sum_log_g += np.log(S_g)

        Z_v = rng.standard_normal(M)
        Z2  = rng.standard_normal(M)
        Z_s = rho*Z_v + np.sqrt(1-rho**2)*Z2

        V_prev = np.maximum(V, 0.0)
        V = (
            V_prev
            + kappa*(theta - V_prev)*dt
            + xi*np.sqrt(V_prev*dt)*Z_v
            + 0.25*xi**2*dt*(Z_v**2 - 1)
        )

        # Milstein‐style log‐Euler for Heston stock
        S_h *= np.exp(
            (r - 0.5*V_prev)*dt
            + np.sqrt(V_prev*dt)*Z_s
            + 0.5*V_prev*dt*(Z_s**2 - 1)
        )

        # plain Euler–Maruyama for GBM control variate
        S_g *= np.exp(
            (r - 0.5*sigma**2)*dt
            + sigma*np.sqrt(dt)*Z_s
        )

    # discrete averages and payoffs
    A = sum_S_h / N
    G = np.exp(sum_log_g / N)
    Y = np.exp(-r*T) * np.maximum(A - K, 0.0)  # arithmetic
    X = np.exp(-r*T) * np.maximum(G - K, 0.0)  # geometric

    # plain MC
    plain_price = np.mean(Y)
    plain_se    = Y.std(ddof=1) / np.sqrt(M)
    plain_var   = Y.var(ddof=1)

    # analytic geometric‐Asian price (control target)
    analy_price = geometric_analytical_asian(S0, K, r, sigma, T, N)

    # control‐variate estimator
    cv_payoffs = Y + c*(analy_price - X)
    cv_price   = np.mean(cv_payoffs)
    cv_se      = cv_payoffs.std(ddof=1) / np.sqrt(M)
    cv_var     = cv_payoffs.var(ddof=1)

    return {
        'plain':       [plain_price, plain_se, plain_var],
        'control_var': [cv_price,     cv_se,     cv_var],
        'analy_price': analy_price,
        'payoffs':     [Y, X]
    }




