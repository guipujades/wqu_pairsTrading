import pandas as pd
from tqdm import tqdm 
import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from datetime import timedelta


def bench_ll(df):
    
    # Get returns and prepare for signal
    df['return'] = df.groupby('Ticker')['Open'].pct_change()
    df = df.dropna(subset=['return'])
    df.set_index('Date', inplace=True)

    period_returns = pd.DataFrame()
    for info, df_stock in tqdm(df.groupby(['Ticker', 'sector'])):
        
        if len(df_stock) > 1:
            returns = df_stock[['Open', 'Ticker', 'return']]
            returns['sector'] = info[1]
            period_returns = pd.concat([period_returns, returns])


    period_returns['year'] = period_returns.index.year
    period_returns['month'] = period_returns.index.month
    period_returns['day'] = period_returns.index.day

    index_monthly_returns = period_returns.groupby(['sector', 'year', 'month'])['return'].mean()

    # Long & Short using Lead-lag signal
    df_ll_prep = period_returns.merge(index_monthly_returns, on=['sector', 'year', 'month'], suffixes=('', '_index'))
    df_ll_prep['period_return'] = df_ll_prep.groupby(['Ticker', 'year', 'month'])['return'].transform(lambda x: (x + 1).cumprod() - 1)

    df_ll_data = []
    for info, df_info in tqdm(df_ll_prep.groupby(['Ticker', 'year', 'month'])):
        df_ll_data.append(df_info.iloc[-1])
    df_ll = pd.DataFrame(df_ll_data)
    df_ll['lead_lag'] = df_ll['period_return'] - df_ll['return_index']

    long_positions = []
    short_positions = []
    for info, df_ in tqdm(df_ll.groupby(['sector', 'year', 'month'])):

        date = list(pd.to_datetime(df_[['year', 'month']].assign(day=1)).dt.to_period('M').unique())[0]
        
        long_pos = df_.loc[df_['lead_lag'].idxmin()]['Ticker']
        short_pos = df_.loc[df_['lead_lag'].idxmax()]['Ticker']
        
        long_positions.append((date, info[0], long_pos))
        short_positions.append((date, info[0], short_pos))
        
    # Build dfs
    short_signal = pd.DataFrame(short_positions, columns=['Date', 'Sector', 'Ticker'])  
    long_signal = pd.DataFrame(long_positions, columns=['Date', 'Sector', 'Ticker'])  

    long_dict = {}
    short_dict = {}
    for date, df in long_signal.groupby('Date'): # inverse
        long_dict[date] = list(df.Ticker)
    for date, df in short_signal.groupby('Date'): # inverse
        short_dict[date] = list(df.Ticker)

    return long_dict, short_dict


def main_strategy(df, df_all_data, test_period):

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
    closest_assets = []
    farthest_assets = []
    for cluster in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans_labels == cluster)[0]
        if len(cluster_indices) > 1:
            closest_index = cluster_indices[np.argmin(distances_to_centroids[cluster_indices, cluster])]
            farthest_index = cluster_indices[np.argmax(distances_to_centroids[cluster_indices, cluster])]
            closest = residuals.columns[closest_index]
            farthest = residuals.columns[farthest_index]
            if closest != farthest:
                operations.append((closest, farthest))
                closest_assets.append((closest, cluster))
                farthest_assets.append((farthest, cluster))
    
    for closest, farthest in operations:
        print(f'Long {closest}, Short {farthest}')
    
    # Plot PCA result with K-Means labels
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=kmeans_labels, palette='tab10')
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
    plt.title('Clusters K-Means (PCA Reduced Data)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()
    
    # Add labels
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=kmeans_labels, palette='tab10')
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
    plt.title('Clusters K-Means (PCA Reduced Data)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    
    # Add labels for closest and farthest assets
    for label, cluster in closest_assets:
        idx = residuals.columns.get_loc(label)
        plt.annotate(label, (pca_result[idx, 0], pca_result[idx, 1]), fontsize=14, alpha=1, color='black', 
                     xytext=(5,5), textcoords='offset points')
    
    for label, cluster in farthest_assets:
        idx = residuals.columns.get_loc(label)
        plt.annotate(label, (pca_result[idx, 0], pca_result[idx, 1]), fontsize=14, alpha=1, color='black', 
                     xytext=(5,5), textcoords='offset points')
    
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
    plt.title('Long-Short vs IBOV')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()