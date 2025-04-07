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


ticker = "AAPL"
start_date = "2010-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")
df = yf.download(ticker, start=start_date, end=end_date)

garman_klass_volatility = calculate_garman_klass_estimator(df)
print(f"Garman-Klass Volatility Estimate: {garman_klass_volatility}")

parkinsons_volatility = calculate_parkinsons_estimator(df)
print(f"Parkinson's Volatility Estimate: {parkinsons_volatility}")

window_size = 30
classic_measures, park_estimates, garman_estimates = get_rolling_window_estimates(df, window_size) 
#plot_rolling_window_estimate(classic_measures, park_estimates, garman_estimates, window_size)

windows = [5, 10, 20, 30, 60, 90]
c_vol_means, p_vol_means, g_vol_means = get_volatility_signature(df, windows)
plot_volatility_signature(windows, c_vol_means, p_vol_means, g_vol_means)



    






