import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# App title and description
st.title("Portfolio Diversification Tool")
st.write("""
This app helps in portfolio diversification by analyzing historical data for various assets, calculating returns, and evaluating risk metrics. 
It provides insights such as portfolio volatility and the Sharpe Ratio, and future updates will include machine learning-driven asset recommendations.
""")

# Select stock tickers dynamically
tickers = st.multiselect(
    'Select stock tickers:', 
    ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'META', 'NFLX'], 
    default=['AAPL', 'AMZN']
)

# Select date range dynamically
start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('2024-01-01'))

# Input risk-free rate
risk_free_rate = st.number_input("Risk-Free Rate (e.g., Treasury bond rate)", min_value=0.0, max_value=1.0, value=0.01)

# Fetch historical data if tickers are selected
if tickers:
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Display data columns
    st.write("Available data columns:", data.columns)

    # Use 'Close' prices for the analysis
    data = data['Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Define equal weights for assets
    weights = np.array([1 / len(tickers)] * len(tickers))  # Equal weights
    
    # Calculate weighted daily returns of the portfolio
    portfolio_returns = returns.dot(weights)
    
    # Calculate portfolio performance metrics
    portfolio_return = portfolio_returns.mean() * 252  # Annualize the return (252 trading days)
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualize volatility
    
    # Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Output portfolio performance and Sharpe Ratio
    st.write(f'Annualized Portfolio Return: {portfolio_return:.2%}')
    st.write(f'Annualized Portfolio Volatility: {portfolio_volatility:.2%}')
    st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    
    # Correlation matrix and heatmap for diversification insights
    st.write("Correlation Matrix:")
    correlation_matrix = returns.corr()
    st.dataframe(correlation_matrix)
    
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(plt)
    
    # Placeholder for future updates
    st.write("Coming soon: Backtesting the model to see historical performance.")
    st.write("Future update: AI-driven asset recommendations based on clustering.")
else:
    st.write("Please select at least one ticker to proceed.")
