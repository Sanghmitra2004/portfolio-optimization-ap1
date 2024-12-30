import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# App title and description
st.title("DiversifyAI")
st.write("""This app helps users build and analyze a diversified investment portfolio using AI-driven asset recommendations. 
The app pulls historical stock data from various assets, calculates their returns, and applies a K-Means clustering algorithm to group assets based on return correlations.
Users can select which stocks to analyze and receive diversification suggestions based on clustering.""")

# Expanded list of stocks from various sectors
tickers = [
    'AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'NFLX', 'SBUX', 'HD', 'FND', 'JCTC',
    'WMT', 'COST', 'MCD', 'BABA', 'BKNG', 'TJX', 'NVDA', 'INTC', 'V', 'MA', 'SPY',
    'XOM', 'DIS', 'GE', 'PFE', 'JNJ', 'UNH', 'VZ', 'CSCO', 'CRM', 'PYPL', 'META',
    'SQ', 'BA', 'UBER', 'LYFT', 'GM', 'F', 'T', 'KO', 'PEP', 'NKE', 'LVMH', 'GS', 'MS'
]

# Select tickers for analysis
selected_tickers = st.multiselect("Select Stocks to Analyze", tickers, default=tickers[:7])

if not selected_tickers:
    st.warning("Please select at least one stock to analyze.")
else:
    # Download historical data for selected tickers
    data = yf.download(selected_tickers, start='2020-01-01', end='2024-01-01')

    # Use 'Close' prices to calculate returns
    data = data['Close']

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Display a sample of the returns data to make sure everything is correct
    st.write("### Daily Returns Sample")
    st.write(returns.head())

    # Define equal weights for selected assets (make sure they sum to 1)
    weights = np.array([1.0 / len(selected_tickers)] * len(selected_tickers))

    # Calculate the weighted daily returns of the portfolio
    portfolio_returns = returns[selected_tickers].dot(weights)

    # Calculate portfolio performance metrics
    portfolio_return = portfolio_returns.mean() * 252  # Annualize the return (252 trading days in a year)
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
    risk_free_rate = 0.01  # Set the risk-free rate (e.g., Treasury bond rate)

    # Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Output portfolio performance and Sharpe Ratio
    st.write(f'### Portfolio Performance')
    st.write(f'Annualized Portfolio Return: {portfolio_return:.2%}')
    st.write(f'Annualized Portfolio Volatility: {portfolio_volatility:.2%}')
    st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    # -------- AI-Driven Asset Recommendation based on Clustering -------- #
    st.write("### AI-Driven Asset Recommendation Based on Clustering")

    # Correlation Matrix for selected assets
    corr_matrix = returns[selected_tickers].corr()

    # Ensure there are no NaN or infinite values in the correlation matrix
    corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    corr_matrix = corr_matrix.fillna(0)  # Replace NaNs with 0

    # Display Correlation Heatmap
    st.write("#### Correlation Heatmap for Selected Assets")
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(plt)

    # Reshape the correlation matrix to be suitable for KMeans
    corr_matrix_values = corr_matrix.values

    # Check if there are enough distinct clusters
    n_clusters = 3  # Change the number of clusters as needed
    if len(selected_tickers) < n_clusters:
        st.warning("You have selected fewer assets than the number of clusters. Adjust the number of clusters or select more assets.")

    else:
        try:
            # Use K-Means Clustering to identify groups of similar assets
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(corr_matrix_values)
            clusters = kmeans.labels_

            # Create a DataFrame for clustering results
            clustered_assets = pd.DataFrame({'Ticker': selected_tickers, 'Cluster': clusters})
            st.write("#### Asset Clusters")
            st.write(clustered_assets)

            # Get the clusters for selected assets
            selected_clusters = clustered_assets['Cluster'].values

            # Ensure that there are enough distinct clusters
            distinct_clusters = set(selected_clusters)

            # If all selected assets belong to the same cluster, show a warning
            if len(distinct_clusters) == 1:
                st.warning("All selected assets belong to the same cluster. We recommend selecting assets from different clusters to improve diversification.")

            # Recommend assets from a different cluster than the selected portfolio
            st.write("#### Recommended Assets for Diversification")
            recommended_assets = clustered_assets[~clustered_assets['Cluster'].isin(selected_clusters)]

            # If no recommended assets exist, display a message
            if recommended_assets.empty:
                st.write("No assets available for diversification. Try selecting a different set of assets.")
            else:
                st.write("Assets for diversification:", recommended_assets['Ticker'].tolist())
        
        except ValueError as e:
            st.error(f"An error occurred during clustering: {str(e)}")
