# Volatility Forecasting with GARCH(1,1)

Forecasts daily volatility of HDFC Bank stock using a GARCH(1,1) model.

## What it does
- Downloads 2 years of NSE price data
- Computes daily percentage returns
- Fits a GARCH(1,1) model to capture volatility clustering
- Compares fitted vs realized volatility and evaluates using MSE
