import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import math
import matplotlib.pyplot as plt

def calculate_mean_estimator(data):
    data['Natural Return'] = (data['Close'].shift(-1) - data['Close']) / data['Close']
    mu_hat = np.mean(data['Natural Return'])
    return mu_hat

def calculate_unbiased_volatility(data, mu_hat=None):
    if mu_hat is None:
        mu_hat = calculate_mean_estimator(data)
    data['Adjusted Return'] = data['Natural Return'] - mu_hat
    data['Squared Adjusted Return'] = data['Adjusted Return'] ** 2
    realized_volatility_estimate = np.sum(data['Squared Adjusted Return'].dropna(), axis=0) / (len(data) - 1)
    return np.sqrt(realized_volatility_estimate)

def get_log_ratios(price1, price2):
    log_high_low_ratio = np.log(price1 / price2)
    return log_high_low_ratio

def calculate_parkinsons_estimator(data):
    log_high_low_ratio = get_log_ratios(data['High'], data['Low'])
    parkinsons_volatility_estimate = np.sqrt((1 / (4 * np.log(2))) * (log_high_low_ratio ** 2).sum())
    return parkinsons_volatility_estimate

def calculate_garman_klass_estimator(data):
    log_high_low_ratio = get_log_ratios(data['High'], data['Low'])
    first_term = (1 / (2 * len(data))) * (log_high_low_ratio ** 2).sum()

    log_close_open_ratio = get_log_ratios(data['Close'], data['Open'])
    second_term = (2 * np.log(2) - 1) * (log_close_open_ratio ** 2).sum() / len(data)

    garman_klass_estimator = np.sqrt(first_term - second_term)
    return garman_klass_estimator

def get_rolling_window_estimates(data, T):
    classic_measures = []
    park_estimates = []
    garman_estimates = []

    # compute volatility for all windows
    for idx in range(len(data) - T + 1):
        current_window = data.iloc[idx : idx + T].copy()
        classic_measures.append(calculate_unbiased_volatility(current_window))
        park_estimates.append(calculate_parkinsons_estimator(current_window))
        garman_estimates.append(calculate_garman_klass_estimator(current_window))

    return classic_measures, park_estimates, garman_estimates

def plot_rolling_window_estimate(classic_measures, park_estimates, garman_estimates, T): 
    # plot volatility estimates
    plt.plot(classic_measures, label='Classic')
    plt.plot(park_estimates, label='Parkinson')
    plt.plot(garman_estimates, label='Garman-Klass')

    plt.title(f"Volatility Estimates for {T} Days")
    plt.xlabel("Days")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

def get_volatility_signature(data, windows):
    classic_measures = []
    park_estimates = []
    garman_estimates = []

    for window in windows:
        classic, park, garman = get_rolling_window_estimates(data, window)
        classic_measures.append(np.mean(classic))
        park_estimates.append(np.mean(park))
        garman_estimates.append(np.mean(garman))

    return classic_measures, park_estimates, garman_estimates

def plot_volatility_signature(windows, avg_classic_list, avg_park_list, avg_garman_list):
    plt.figure(figsize=(10, 6))
    plt.plot(windows, avg_classic_list, marker='o', label='Classic')
    plt.plot(windows, avg_park_list, marker='o', label='Parkinson')
    plt.plot(windows, avg_garman_list, marker='o', label='Garman-Klass')
    plt.xlabel('Window Size (m)')
    plt.ylabel('Average Realized Volatility')
    plt.title('Volatility Signature Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_closest_expiry(spx_symbol, today, days_from_today=30):
    spx_ticker = yf.Ticker(spx_symbol)
    expiry_dates = spx_ticker.options
    if today is None:
        today = datetime.datetime.today()
    expiry_dates_sorted = sorted([datetime.datetime.strptime(expiry, "%Y-%m-%d") for expiry in expiry_dates])
    closest_expiry = None
    min_diff = float('inf')
    for expiry_date in expiry_dates_sorted:
        days_to_expiry = (expiry_date - today).days
        if days_to_expiry >= 0:
            diff = abs(days_to_expiry - days_from_today)
            if diff < min_diff:
                min_diff = diff
                closest_expiry = expiry_date
    return closest_expiry

def calculate_calls_puts_sum(puts_df, calls_df):
    vix_sum = 0
    nputs = len(puts_df)
    for i in range(nputs - 1):
        Kp_i = puts_df.iloc[i]['strike']
        Kp_i_next = puts_df.iloc[i + 1]['strike']
        P_i = puts_df.iloc[i]['lastPrice']
        vix_sum += P_i * (1 / Kp_i - 1 / Kp_i_next)
    
    np_calls = len(calls_df)
    for i in range(1, np_calls):
        Kc_i = calls_df.iloc[i]['strike']
        Kc_i_prev = calls_df.iloc[i - 1]['strike'] if i > 0 else Kc_i  
        C_i = calls_df.iloc[i]['lastPrice']
        vix_sum += C_i * (1 / Kc_i_prev - 1 / Kc_i)
    
    return vix_sum

def VIX_estimator(puts_df, calls_df, F0, r=0.02, tau=41/365):
    vix_sum = calculate_calls_puts_sum(puts_df, calls_df)
    vix_square = (2 * np.exp(r * tau) / tau) * vix_sum
    return np.sqrt(vix_square) * 100      






