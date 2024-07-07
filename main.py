import tempfile
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm 
import statsmodels.api as sm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from datetime import timedelta

from data_handler import *
from data_utils import *

# Get data
tmp_path = tempfile.mkdtemp()
t = Tickers_B3(tmp_path)
t.open_file()
assets, prices_enfoque, sectors_b3 = t.get_tickers()

# Get benchmark data
start_date = prices_enfoque.Date.unique()[0]
final_date = prices_enfoque.Date.unique()[-1]
ibov = yf.download('^BVSP', start=start_date, end=final_date)
ibov.index = pd.to_datetime(ibov.index)

# Sector filter
prices_enfoque['ticker_s'] = prices_enfoque['Ticker'].str.strip().str[0:4]
prices_enfoque.set_index('ticker_s', inplace=True)
sectors_b3.set_index('ticker', inplace=True)
all_sectors = list(sectors_b3['sector'].unique()) # we're using all sectors
sectors_b3 = sectors_b3[sectors_b3['sector'].isin(all_sectors)]

# Prepare returns
df = prices_enfoque.join(sectors_b3['sector'], how='left')
df['Date'] = pd.to_datetime(df['Date'])
df = df.drop_duplicates()
df.set_index('Date', inplace=True)

# Check df dates integrity
df = df[df.index.isin(ibov.index)]

# Merge returns
df['return'] = df.groupby('Ticker')['Open'].transform(lambda x: np.log(x).diff())
ibov['ibov_returns'] = np.log(ibov['Adj Close']/ibov['Adj Close'].shift(1))
df = pd.merge(df, ibov['ibov_returns'], on='Date')

# Test period (only one run): pre-backtesting
df_all_data = df.copy()
test_period = pd.to_datetime('2010-03-01')
df = df[df.index <= test_period]

# Liquidity filter
stock_reference = 'PETR4'
liquidity_filter = pd.DataFrame({'Vol_filter': df[df.Ticker==stock_reference]['Financial_Volume'].rolling(21).mean()})

# Apply liquidity: this is just to avoid stocks with no liquidity at all
df['rolling_mean'] = df.groupby('Ticker')['Financial_Volume'].transform(lambda x: x.rolling(window=21).mean())
df = pd.merge(df.reset_index(), liquidity_filter, on=['Date'], how='inner')
liquidity_mask = df['rolling_mean'] < df['Vol_filter'] * 0.01 # careful not to change df size
date_backup = df['Date'].copy()
ticker_backup = df['Ticker'].copy()

df.loc[liquidity_mask, :] = np.nan
df['Date'] = date_backup
df['Ticker'] = ticker_backup
df.set_index('Date', inplace=True)

assets = df['Ticker'].unique()
residuals = pd.DataFrame(index=df.index.unique())
for ticker in tqdm(assets):
    
    df_ = df[df['Ticker'] == ticker]
    df_ = df_[['return', 'ibov_returns']].dropna() 
    if len(df_) == 0:
        continue
    
    # Deal with nans (liquidity filter)
    min_non_na = int(0.9 * len(df_))
    df_.dropna(thresh=min_non_na, axis=1, inplace=True)
    
    # Get residuals
    X = sm.add_constant(df_['ibov_returns'])
    y = df_['return']
    model = sm.OLS(y, X).fit()
    resid = model.resid
    residuals[ticker] = resid

# Deal with nans
min_non_na = int(0.9 * len(residuals))
residuals.dropna(thresh=min_non_na, axis=1, inplace=True)
residuals = residuals.ffill().bfill()

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(residuals.T)

# Optimal number of cluster
inertia = []
silhouette_scores = []
K = range(2, 11)  # test between 2 to 10 clusters
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_result)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(pca_result, kmeans.labels_)
    silhouette_scores.append(score)

# OPtimal number results
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores For Optimal k')

plt.tight_layout()
plt.show()

# Best number of cluster
optimal_k = K[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_result)

# Distances to centroids for each cluster
distances_to_centroids = np.zeros((pca_result.shape[0], kmeans.n_clusters))
for i in range(kmeans.n_clusters):
    distances_to_centroids[:, i] = np.linalg.norm(pca_result - kmeans.cluster_centers_[i], axis=1)

# Operations
operations = []

# Select stocks for long and short within each cluster
for cluster in range(kmeans.n_clusters):
    cluster_indices = np.where(kmeans_labels == cluster)[0]
    
    if len(cluster_indices) > 1:  
        
        closest_index = cluster_indices[np.argmin(distances_to_centroids[cluster_indices, cluster])]
        farthest_index = cluster_indices[np.argmax(distances_to_centroids[cluster_indices, cluster])]

        closest = residuals.columns[closest_index]
        farthest = residuals.columns[farthest_index]

        if closest != farthest:
            operations.append((closest, farthest))

for closest, farthest in operations:
    print(f'Long {closest}, Short {farthest}')

# Plot PCA result with K-Means labels
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=kmeans_labels, palette='tab10')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('Clusters K-Means (PCA Reduced Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Simple backtest
def calculate_cumulative_return(prices):
    returns = np.log(prices / prices.shift(1))
    cumulative_return = np.exp(returns.cumsum()) - 1
    return cumulative_return

# PerIod
start_test_period = test_period + timedelta(days=1)
end_test_period = start_test_period + timedelta(days=90)

selected_tickers = [str(ticker) for pair in operations for ticker in pair]
df_all_data = df_all_data[df_all_data.Ticker.isin(selected_tickers)]
df_all_data = df_all_data[(df_all_data.index>=start_test_period) & (df_all_data.index<=end_test_period)]

df_bt = df_all_data[['Ticker', 'return', 'ibov_returns']]
df_bt.reset_index(inplace=True)
df_bt.set_index(['Date', 'Ticker'], inplace=True)

# Cumulative returns
stocks_cumulative_returns = df_bt.groupby('Ticker')['return'].cumsum()
stocks_cumulative_returns = np.exp(stocks_cumulative_returns) - 1
df_bt['cumulative_return'] = stocks_cumulative_returns

# COlumn as stocks
df_str_bt = df_bt.reset_index().pivot(index='Date', columns='Ticker', values='cumulative_return')

# Ibov returns
ibov_returns = df_bt.groupby('Date')['ibov_returns'].sum()
ibov_cumulative_returns = np.exp(ibov_returns.cumsum()) - 1
df_str_bt['ibov'] = ibov_cumulative_returns

# Long-short: 1:1 leverage
strategies_cumulative_returns = pd.DataFrame(index=df_str_bt.index)
for long_stock, short_stock in operations:
    strategies_cumulative_returns[f'Long {long_stock}, Short {short_stock}'] = df_str_bt[long_stock] - df_str_bt[short_stock]

# Results
plt.figure(figsize=(14, 8))
for strategy in strategies_cumulative_returns.columns:
    plt.plot(strategies_cumulative_returns[strategy], label=strategy, linestyle='--')

plt.plot(ibov_cumulative_returns.index, ibov_cumulative_returns, label='IBOV', linewidth=2)
plt.title('Retornos Acumulados: Estratégias Long-Short vs IBOV')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.legend()
plt.grid(True)
plt.show()


