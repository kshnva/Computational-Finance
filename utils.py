import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import math

def calculate_mean_estimator(data):
    data['Natural Return'] = (data['Close'].shift(-1) - data['Close']) / data['Close']
    mu_hat = np.mean(data['Natural Return'])
    return mu_hat

def calculate_unbiased_volatility(data, mu_hat=None):
    if mu_hat is None:
        mu_hat = calculate_mean_estimator(data)
    data['Adjusted Return'] = data['Natural Return'] - mu_hat
    data['Squared Adjusted Return'] = data['Adjusted Return'] ** 2
    realized_volatility_estimate = np.sum(data['Squared Adjusted Return'].dropna()) / (len(data) - 1)
    return np.sqrt(realized_volatility_estimate)

def get_log_ratios(price1, price2):
    log_high_low_ratio = np.log(price1 / price2)
    return log_high_low_ratio

def calculate_parkinsons_estimator(data):
    log_high_low_ratio = get_log_ratios(data['High'], data['Low'])
    parkinsons_volatility_estimate = np.sqrt((1 / (4 * np.log(2))) * np.sum(log_high_low_ratio ** 2))
    return parkinsons_volatility_estimate

def calculate_garman_klass_estimator(data):
    log_high_low_ratio = get_log_ratios(data['High'], data['Low'])
    first_term = (1 / (2 * len(data))) * np.sum(log_high_low_ratio ** 2)

    log_close_open_ratio = get_log_ratios(data['Close'], data['Open'])
    second_term = (2 * np.log(2) - 1) * np.sum(log_close_open_ratio ** 2) / len(data)

    garman_klass_estimator = np.sqrt(first_term - second_term)
    return garman_klass_estimator

ticker = "AAPL"
start_date = "2010-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")
data = yf.download(ticker, start=start_date, end=end_date)

garman_klass_volatility = calculate_garman_klass_estimator(data)
print(f"Garman-Klass Volatility Estimate: {garman_klass_volatility}")

parkinsons_volatility = calculate_parkinsons_estimator(data)
print(f"Parkinson's Volatility Estimate: {parkinsons_volatility}")
