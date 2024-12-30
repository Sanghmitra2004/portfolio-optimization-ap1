import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd

# Define the stock tickers you want to analyze
tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']  # Replace with your asset list

# Download historical data (adjust as needed)
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')

# Check and display the column names to understand the data
st.write(data.columns)

# Use 'Close' prices to calculate returns
data = data['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Define weights for each asset (make sure they sum to 1)
weights = np.array([0.2] * len(tickers))  # Equal weights for 5 assets (0.2 each)

# Calculate the weighted daily returns of the portfolio
portfolio_returns = returns.dot(weights)

# Calculate portfolio performance metrics
portfolio_return = portfolio_returns.mean() * 252  # Annualize the return (252 trading days in a year)
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
risk_free_rate = 0.01  # Set the risk-free rate (e.g., Treasury bond rate)

# Calculate Sharpe Ratio
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Output portfolio performance and Sharpe Ratio
st.write(f'Annualized Portfolio Return: {portfolio_return:.2%}')
st.write(f'Annualized Portfolio Volatility: {portfolio_volatility:.2%}')
st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')
