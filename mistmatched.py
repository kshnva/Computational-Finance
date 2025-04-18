import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import hedging as hedge 

def run_mismatched_volatility_simulation(S0, K, r, sigma_model, sigma_market, T, hedge_frequency=1, num_paths=10000, seed=42):
    np.random.seed(seed)
    dt, N = hedge.setup_simulation_parameters(T)
    initial_price = hedge.bs_call_price(S0, K, r, sigma_model, T)
    initial_delta = hedge.bs_call_delta(S0, K, r, sigma_model, T)

    pnl_list = []
    for _ in range(num_paths):
        S = S0
        stock_position = initial_delta
        cash = -S * stock_position + initial_price

        for day in range(1, N + 1):
            dW = np.random.normal(0, 1) * np.sqrt(dt)
            S = S * (1 + r * dt + sigma_market * dW)
            cash *= np.exp(r * dt)

            if day % hedge_frequency == 0 and day < N:
                tau = T - day * dt
                new_delta = hedge.bs_call_delta(S, K, r, sigma_model, tau)
                cash -= S * (new_delta - stock_position)
                stock_position = new_delta

            if day == N:
                option_payoff = max(0, S - K)
                final_value = cash + S * stock_position - option_payoff
                pnl_list.append(final_value)

    return pd.DataFrame(pnl_list, columns=['PnL'])

def create_summary_statistics(results):
    summary = pd.DataFrame({
        'Scenario': list(results.keys()),
        'Mean PnL': [results[k]['PnL'].mean() for k in results],
        'Std Dev PnL': [results[k]['PnL'].std() for k in results],
        'Min PnL': [results[k]['PnL'].min() for k in results],
        'Max PnL': [results[k]['PnL'].max() for k in results],
    })
    return summary

def plot_density(results, title):
    plt.figure(figsize=(14, 8))
    for scenario in results:
        results[scenario]['PnL'].plot.density(label=scenario)
    plt.title(title)
    plt.xlabel('P&L (EUR)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha =0.3)
    plt.show()

def plot_histograms(results, title):
    plt.figure(figsize=(14, 8))
    for scenario in results:
        plt.hist(results[scenario]['PnL'], bins=50, alpha=0.5, label=scenario)
    plt.title(title)
    plt.xlabel('P&L (EUR)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha =0.3)
    plt.show()

def plot_boxplots(results, title):
    plt.figure(figsize=(14, 8))
    boxplot_data = [results[scenario]['PnL'] for scenario in results]
    plt.boxplot(boxplot_data, labels=list(results.keys()))
    plt.title(title)
    plt.ylabel('P&L (EUR)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, alpha =0.3)
    plt.show()

def run_hedging_scenarios(S0, K, r, T, sigma_model, market_volatilities, hedge_frequency):
    results = {}
    for sigma_market in market_volatilities:
        scenario = f"Market Vol {sigma_market*100:.0f}% vs Model Vol {sigma_model*100:.0f}%"
        results[scenario] = run_mismatched_volatility_simulation(
            S0, K, r, sigma_model, sigma_market, T, hedge_frequency)
    return results

def plot_mean_std_vs_volatility(vol_values, mean_pnl_daily, mean_pnl_weekly, std_pnl_daily, std_pnl_weekly, sigma_model):
    plt.figure(figsize=(12, 6))
    plt.plot(vol_values, mean_pnl_daily, 'o-', label='Daily Hedging')
    plt.plot(vol_values, mean_pnl_weekly, 's-', label='Weekly Hedging')
    plt.axvline(x=sigma_model, color='r', linestyle='--', label=f'Model Volatility ({sigma_model*100:.0f}%)')
    plt.title('Impact of Market Volatility on Mean P&L')
    plt.xlabel('Actual Market Volatility')
    plt.ylabel('Mean P&L (EUR)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(vol_values, std_pnl_daily, 'o-', label='Daily Hedging')
    plt.plot(vol_values, std_pnl_weekly, 's-', label='Weekly Hedging')
    plt.axvline(x=sigma_model, color='r', linestyle='--', label=f'Model Volatility ({sigma_model*100:.0f}%)')
    plt.title('Impact of Market Volatility on P&L Standard Deviation')
    plt.xlabel('Actual Market Volatility')
    plt.ylabel('P&L Standard Deviation (EUR)')
    plt.legend()
    plt.grid(True)
    plt.show()
