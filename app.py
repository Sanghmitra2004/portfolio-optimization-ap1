import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd

# Define the stock tickers you want to analyze
tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']  # Replace with your asset list

# Download historical data (adjust as needed)
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')

# Print the column names to check what's available
st.write(data.columns)

# Use 'Close' prices if 'Adj Close' is unavailable
data = data['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Define weights for each asset (make sure they sum to 1)
weights = np.array([0.04] * len(tickers))  # Example with equal weights

# Calculate the weighted daily returns
portfolio_returns = returns.dot(weights)

# Calculate portfolio performance metrics
portfolio_return = portfolio_returns.mean() * 252  # Annualize the return (252 trading days in a year)
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
risk_free_rate = 0.01  # Set the risk-free rate (e.g., Treasury bond rate)
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Output Sharpe Ratio
st.write(f'Sharpe Ratio: {sharpe_ratio}')
