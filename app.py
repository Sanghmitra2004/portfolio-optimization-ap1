import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
#This app is designed to help users build and analyze a diversified investment portfolio by using an AI-driven approach to asset recommendations.
#The app pulls historical stock data from various assets, calculates their returns, and applies a machine learning clustering algorithm (K-Means) to group the assets based on their return correlations.
st.title(" DiversifyAI")
st.write("""This app is designed to help users build and analyze a diversified investment portfolio by using an AI-driven approach to asset recommendations. 
The app pulls historical stock data from various assets, calculates their returns, and applies a machine learning clustering algorithm (K-Means) to group the assets based on their return correlations.""")
# Define the stock tickers you want to analyze
tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'NFLX', 'SBUX', 'HD', 'FND', 'JCTC', 'WMT', 'COST', 'MCD', 'BABA', 'BKNG', 'TJX']  # Replace with your asset list

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

# -------- AI-Driven Asset Recommendation based on Clustering -------- #
st.write("### AI-Driven Asset Recommendation Based on Clustering")

# Correlation Matrix
corr_matrix = returns.corr()

# Display Correlation Heatmap
st.write("#### Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(plt)

# Use K-Means Clustering to identify groups of similar assets
kmeans = KMeans(n_clusters=3)  # Change the number of clusters as needed
kmeans.fit(corr_matrix)
clusters = kmeans.labels_

# Display Clustering Results
clustered_assets = pd.DataFrame({'Ticker': tickers, 'Cluster': clusters})
st.write("#### Asset Clusters")
st.write(clustered_assets)

# Recommend assets from a different cluster than current portfolio
st.write("#### Recommended Assets for Diversification")
current_cluster = clusters[0]  # Assume we're starting with the first asset's cluster
recommended_assets = clustered_assets[clustered_assets['Cluster'] != current_cluster]
st.write(recommended_assets['Ticker'].tolist())  # Display the recommended assets

