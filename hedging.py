import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
# Black-Scholes formulas
def d1(S, K, r, sigma, tau):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def d2(S, K, r, sigma, tau):
    return d1(S, K, r, sigma, tau) - sigma * np.sqrt(tau)

def bs_call_price(S, K, r, sigma, tau):
    if tau <= 0:
        return max(0, S - K)
    return S * norm.cdf(d1(S, K, r, sigma, tau)) - K * np.exp(-r * tau) * norm.cdf(d2(S, K, r, sigma, tau))

def bs_call_delta(S, K, r, sigma, tau):
    if tau <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, r, sigma, tau))

def setup_simulation_parameters(T, days_per_year=252):
    """Set up basic time parameters for the simulation"""
    dt = 1/days_per_year
    N = int(T * days_per_year)
    return dt, N

def initialize_portfolio(S0, K, r, sigma, T):
    """Initialize portfolio values at the start of simulation"""
    initial_price = bs_call_price(S0, K, r, sigma, T)
    initial_delta = bs_call_delta(S0, K, r, sigma, T)
    portfolio = {
        'cash': -S0 * initial_delta + initial_price,
        'stock_position': initial_delta,
        'portfolio_value': -initial_price
    }
    return portfolio

def update_stock_price(S, r, sigma, dt):
    """Generate a new stock price for the next time step"""
    dW = np.random.normal(0, 1) * np.sqrt(dt)
    return S * (1 + r * dt + sigma * dW)

def rebalance_portfolio(S, K, r, sigma, tau, portfolio):
    """Rebalance portfolio based on new delta"""
    new_delta = bs_call_delta(S, K, r, sigma, tau)
    portfolio['cash'] -= S * (new_delta - portfolio['stock_position'])
    portfolio['stock_position'] = new_delta
    return portfolio

def calculate_final_pnl(S, K, portfolio):
    """Calculate the final P&L at option expiry"""
    option_payoff = max(0, S - K)
    return portfolio['cash'] + S * portfolio['stock_position'] - option_payoff

def run_single_path(S0, K, r, sigma, T, hedge_frequency, dt, N):
    """Simulate a single path and return the final P&L"""
    portfolio = initialize_portfolio(S0, K, r, sigma, T)
    S = S0
    
    for day in range(1, N+1):
        # Update stock price
        S = update_stock_price(S, r, sigma, dt)
        
        # Update cash position with interest
        portfolio['cash'] *= np.exp(r * dt)
        
        # Rebalance if needed
        if day % hedge_frequency == 0 and day < N:
            tau = T - day * dt
            portfolio = rebalance_portfolio(S, K, r, sigma, tau, portfolio)
        
        # Calculate final P&L at expiry
        if day == N:
            return calculate_final_pnl(S, K, portfolio)
    
    return None  # Should never reach here

def run_hedging_simulation(S0, K, r, sigma, T, hedge_frequency, num_paths=10000, seed=42):
    """Run multiple simulation paths and collect P&L results"""
    np.random.seed(seed)
    dt, N = setup_simulation_parameters(T)
    
    pnl_list = []
    for path in range(num_paths):
        final_pnl = run_single_path(S0, K, r, sigma, T, hedge_frequency, dt, N)
        pnl_list.append(final_pnl)
    
    return pd.DataFrame(pnl_list, columns=['PnL'])
def run_simulations_with_different_frequencies(base_params, hedge_frequencies=None):
    """Run hedging simulations with different hedging frequencies."""
    if hedge_frequencies is None:
        hedge_frequencies = [1, 5, 21]  # Default: Daily, Weekly, Monthly
    
    results = {}
    for freq in hedge_frequencies:
        freq_name = {1: "Daily", 5: "Weekly", 21: "Monthly"}.get(freq, f"Every {freq} days")
        results[freq_name] = run_hedging_simulation(
            base_params['S0'], 
            base_params['K'], 
            base_params['r'], 
            base_params['sigma'], 
            base_params['T'], 
            freq
        )
    
    return results

def create_summary_statistics(results):
    """Create a summary dataframe with key statistics for each simulation."""
    summary = pd.DataFrame({
        'Hedging Frequency': list(results.keys()),
        'Mean PnL': [results[freq]['PnL'].mean() for freq in results],
        'Std Dev PnL': [results[freq]['PnL'].std() for freq in results],
        'Min PnL': [results[freq]['PnL'].min() for freq in results],
        'Max PnL': [results[freq]['PnL'].max() for freq in results],
    })
    
    return summary

def plot_histogram_distribution(results, title='P&L Distribution by Hedging Frequency'):
    """Create histogram plots for P&L distributions."""
    plt.figure(figsize=(12, 8))
    for freq in results:
        plt.hist(results[freq]['PnL'], bins=50, alpha=0.5, label=f"{freq} Hedging")
    
    plt.title(title)
    plt.xlabel('P&L (EUR)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()  # Return the current figure

def plot_density_curves(results, title='P&L Density by Hedging Frequency'):
    """Create density plots for P&L distributions."""
    plt.figure(figsize=(12, 8))
    for freq in results:
        results[freq]['PnL'].plot.density(label=f"{freq} Hedging")
    
    plt.title(title)
    plt.xlabel('P&L (EUR)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()  # Return the current figure

def plot_boxplot_comparison(results, title='P&L Distribution by Hedging Frequency'):
    """Create boxplot comparison of P&L distributions."""
    plt.figure(figsize=(10, 6))
    boxplot_data = [results[freq]['PnL'] for freq in results]
    
    plt.boxplot(boxplot_data, labels=list(results.keys()))
    plt.title(title)
    plt.ylabel('P&L (EUR)')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()  # Return the current figure

def run_standard_hedging_analysis(base_params=None, hedge_frequencies=None, show_plots=True):
    """Run the complete standard hedging analysis with customizable parameters."""
    # Use default parameters if none provided
    if base_params is None:
        base_params = {
            'S0': 100,  # Initial stock price
            'K': 99,    # Strike price
            'r': 0.06,  # Risk-free rate
            'sigma': 0.20,  # Volatility
            'T': 1      # Time to maturity
        }
    
    # Run simulations
    results = run_simulations_with_different_frequencies(base_params, hedge_frequencies)
    
    # Create summary statistics
    summary = create_summary_statistics(results)
 
    print("Summary Statistics for Hedging Simulation:")
    print(summary)
    
    # Generate and show plots if requested
    if show_plots:
        plot_histogram_distribution(results)
        plot_density_curves(results)
        plot_boxplot_comparison(results)
    
    return summary, results

hedge_frequencies = [1, 5, 21]
