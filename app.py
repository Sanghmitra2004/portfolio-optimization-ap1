import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# This app helps users build and analyze a diversified investment portfolio using AI-driven asset recommendations.
st.title("DiversifyAI")
st.write("""This app helps users build and analyze a diversified investment portfolio using AI-driven asset recommendations. 
The app pulls historical stock data from various assets, calculates their returns, and applies a K-Means clustering algorithm to group assets based on return correlations. 
Users can select which stocks to analyze and receive diversification suggestions based on clustering.""")

# Define the stock tickers you want to analyze
tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'NFLX', 'SBUX', 'HD', 'FND', 'JCTC', 'WMT', 'COST', 'MCD', 'BABA', 'BKNG', 'TJX']

# Download historical data (adjust as needed)
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')

# Extract 'Close' prices and calculate daily returns
data = data['Close']
returns = data.pct_change().dropna()

# Create a correlation matrix
corr_matrix = returns.corr()

# Perform K-Means Clustering on the entire set of assets
kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
kmeans.fit(corr_matrix)
clusters = kmeans.labels_

# Display the correlation heatmap for all assets
st.write("### Correlation Heatmap for All Assets")
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(plt)

# Display clusters for all assets
clustered_assets = pd.DataFrame({'Ticker': tickers, 'Cluster': clusters})
st.write("### Asset Clusters")
st.write(clustered_assets)

# --- Portfolio selection section ---
# Let the user select stocks for their portfolio
selected_stocks = st.multiselect("Select Stocks to Analyze", tickers)

if selected_stocks:
    # Filter returns for selected stocks
    selected_data = data[selected_stocks]
    selected_returns = selected_data.pct_change().dropna()

    # Portfolio Performance Metrics
    weights = np.array([1/len(selected_stocks)] * len(selected_stocks))  # Equal weight for each selected stock
    portfolio_returns = selected_returns.dot(weights)

    portfolio_return = portfolio_returns.mean() * 252  # Annualized return
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
    risk_free_rate = 0.01  # Risk-free rate (e.g., Treasury bond rate)

    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Display portfolio performance metrics
    st.write(f'Annualized Portfolio Return: {portfolio_return:.2%}')
    st.write(f'Annualized Portfolio Volatility: {portfolio_volatility:.2%}')
    st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    # --- AI-driven Asset Recommendation Based on Clustering ---
    st.write("### AI-Driven Asset Recommendation Based on Clustering")

    # Recommend assets from the entire set excluding the selected ones
    available_assets = [asset for asset in tickers if asset not in selected_stocks]

    if available_assets:
        st.write("#### Recommended Assets for Diversification")
        st.write(available_assets)  # Display the recommended assets

    # Correlation heatmap for the selected stocks
    st.write("### Correlation Heatmap for Selected Assets")
    selected_corr_matrix = selected_returns.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(plt)

else:
    st.write("Please select stocks to analyze.")
