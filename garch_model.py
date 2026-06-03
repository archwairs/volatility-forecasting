"""
Volatility Forecasting using GARCH(1,1)
Stock: HDFC Bank (NSE)
Model: GARCH(1,1) — captures volatility clustering in daily returns
"""

import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
from sklearn.metrics import mean_squared_error  

hdfc = yf.download("HDFCBANK.NS", start = "2024-01-01", end = "2026-01-01")["Close"].squeeze()

# Calculating 1-Day returns
returns = hdfc.pct_change().dropna()*100

# Fitting the GARCH(1,1) model
model = arch_model(returns, vol='GARCH', p = 1, q = 1)
result = model.fit(disp='off')

fitted_vol = result.conditional_volatility

forecast = result.forecast(horizon=1)
next_day_vol = forecast.variance.values[-1, 0] ** 0.5

realized_vol = returns.rolling(window=21).std()

common_index = fitted_vol.index.intersection(realized_vol.dropna().index)
mse = mean_squared_error(realized_vol.loc[common_index], fitted_vol.loc[common_index])

print(f"Stock: HDFC Bank")
print(f"Forecasted volatility for next trading day: {next_day_vol:.4f}%")
print(f"MSE (GARCH vs Realized): {mse:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(realized_vol, label="Realized Volatility", alpha=0.7)
plt.plot(fitted_vol, label="GARCH Fitted Volatility", alpha=0.7)
plt.legend()
plt.title("Realized vs GARCH Fitted Volatility")
plt.ylabel("Volatility (%)")
plt.show()
