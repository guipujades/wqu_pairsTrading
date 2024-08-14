import os
os.environ["OMP_NUM_THREADS"] = "1"

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
from sklearn.cluster import DBSCAN


def bench_ll(df, frequency='monthly'):
    
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

    # Determina o período de agrupamento com base na frequência
    if frequency == 'monthly':
        period_returns['year'] = period_returns.index.year
        period_returns['month'] = period_returns.index.month
        group_by_cols = ['sector', 'year', 'month']
        index_returns = period_returns.groupby(group_by_cols)['return'].mean()
    elif frequency == 'weekly':
        period_returns['year'] = period_returns.index.year
        period_returns['week'] = period_returns.index.isocalendar().week
        group_by_cols = ['sector', 'year', 'week']
        index_returns = period_returns.groupby(group_by_cols)['return'].mean()

    # Long & Short using Lead-lag signal
    df_ll_prep = period_returns.merge(index_returns, on=group_by_cols, suffixes=('', '_index'))
    df_ll_prep['period_return'] = df_ll_prep.groupby(['Ticker'] + group_by_cols[1:])['return'].transform(lambda x: (x + 1).cumprod() - 1)

    df_ll_data = []
    for info, df_info in tqdm(df_ll_prep.groupby(['Ticker'] + group_by_cols[1:])):
        df_ll_data.append(df_info.iloc[-1])
    df_ll = pd.DataFrame(df_ll_data)
    df_ll['lead_lag'] = df_ll['period_return'] - df_ll['return_index']
    
    n_long = 2  # n long
    n_short = 2  # n short
    
    long_positions = []
    short_positions = []
    for info, df_ in tqdm(df_ll.groupby(group_by_cols)):

        # Ajuste da data com base na frequência
        if frequency == 'monthly':
            date = list(pd.to_datetime(df_[['year', 'month']].assign(day=1)).dt.to_period('M').unique())[0]
        elif frequency == 'weekly':
            date = pd.to_datetime(f'{info[1]}{info[2]}0', format='%Y%W%w') + pd.offsets.Week(weekday=6)
        
        long_pos = df_.nsmallest(n_long, 'lead_lag')['Ticker'].tolist()
        short_pos = df_.nlargest(n_short, 'lead_lag')['Ticker'].tolist()

        for ticker in long_pos:
            short_positions.append((date, info[0], ticker))
        for ticker in short_pos:
            long_positions.append((date, info[0], ticker))
        
    # Build dfs
    short_signal = pd.DataFrame(short_positions, columns=['Date', 'Sector', 'Ticker'])  
    long_signal = pd.DataFrame(long_positions, columns=['Date', 'Sector', 'Ticker'])  

    long_dict = {}
    short_dict = {}
    for date, df in long_signal.groupby('Date'):
        long_dict[date] = list(df.Ticker)
    for date, df in short_signal.groupby('Date'): 
        short_dict[date] = list(df.Ticker)

    return long_dict, short_dict


def main_strategy_kmeans(df, start_period, lookback_period=365):
    
    df.set_index('Date', inplace=True)
    
    end_period = df.index.max() 
    long_dict = {}
    short_dict = {}
    current_period = start_period
    first_iteration = True
    
    while current_period <= end_period:
        print(f"Running strategy for {current_period.date()}")
        
        # Lookback period
        lookback_start = current_period - timedelta(days=lookback_period)
        
        # Lookback filter
        df_lookback = df[(df.index >= lookback_start) & (df.index <= current_period)]
        
        assets = df_lookback['Ticker'].unique()
        residuals = pd.DataFrame(index=df_lookback.index.unique())
        residuals_list = []
        for ticker in tqdm(assets):
            df_ticker = df_lookback[df_lookback['Ticker'] == ticker]
            df_ticker = df_ticker[['return', 'ibov_returns']].dropna() 
            if len(df_ticker) == 0:
                continue
            
            # Residuals
            X = sm.add_constant(df_ticker['ibov_returns'])
            y = df_ticker['return']
            model = sm.OLS(y, X).fit()
            resid = model.resid
            # residuals[ticker] = resid
            residuals_list.append(pd.DataFrame({ticker: resid}))
            
        if residuals_list:
            residuals = pd.concat(residuals_list, axis=1)
        
        # Deal with NaNs
        min_non_na = int(0.9 * len(residuals))
        residuals.dropna(thresh=min_non_na, axis=1, inplace=True)
        residuals = residuals.ffill().bfill()
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(residuals.T)
        
        # Cls number
        silhouette_scores = []
        K = range(2, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pca_result)
            score = silhouette_score(pca_result, kmeans.labels_)
            silhouette_scores.append(score)
        
        optimal_k = K[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(pca_result)
        
        # Dist
        distances_to_centroids = np.zeros((pca_result.shape[0], kmeans.n_clusters))
        for i in range(kmeans.n_clusters):
            distances_to_centroids[:, i] = np.linalg.norm(pca_result - kmeans.cluster_centers_[i], axis=1)
        
        # ID positions
        long_positions = []
        short_positions = []
        for cluster in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans_labels == cluster)[0]
            if len(cluster_indices) > 1:
                
                sorted_indices = np.argsort(distances_to_centroids[cluster_indices, cluster])
                n_long = min(15, len(sorted_indices))  
                n_short = min(15, len(sorted_indices)) 
                
                for i in range(n_long):
                    long_positions.append(residuals.columns[sorted_indices[i]])
                
                for i in range(-n_short, 0):  # Escolhe os mais distantes para short
                    short_positions.append(residuals.columns[sorted_indices[i]])
                
                
                # closest_index = cluster_indices[np.argmin(distances_to_centroids[cluster_indices, cluster])]
                # farthest_index = cluster_indices[np.argmax(distances_to_centroids[cluster_indices, cluster])]
                # closest = residuals.columns[closest_index]
                # farthest = residuals.columns[farthest_index]
                # if closest != farthest:
                #     long_positions.append(closest)
                #     short_positions.append(farthest)
        
        long_dict[current_period] = long_positions
        short_dict[current_period] = short_positions
        
        if first_iteration:
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
            plt.title(f"K-Means Clustering (k={optimal_k}) - {current_period.date()}")
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.colorbar(label='Cluster Label')
            first_iteration = False
            
            # Id pos
            for i, ticker in enumerate(residuals.columns):
                if ticker in long_positions:
                    plt.annotate(ticker, (pca_result[i, 0], pca_result[i, 1]), color='blue', fontsize=8, fontweight='bold')
                elif ticker in short_positions:
                    plt.annotate(ticker, (pca_result[i, 0], pca_result[i, 1]), color='red', fontsize=8, fontweight='bold')
            
            plt.show()
        
        # Next month
        current_period += pd.offsets.MonthBegin(1)
        # current_period += pd.offsets.Week(1)

    return long_dict, short_dict


def main_strategy_dbscan(df, start_period, lookback_period=365, db_eps=0.3, db_min_samples=6, turn_period='monthly'):
    
    df.set_index('Date', inplace=True)

    end_period = df.index.max() 
    long_dict = {}
    short_dict = {}
    current_period = start_period
    first_iteration = True
    
    while current_period <= end_period:
        print(f"Running strategy for {current_period.date()}")
        
        # Lookback period
        lookback_start = current_period - timedelta(days=lookback_period)
        
        # Lookback filter
        df_lookback = df[(df.index >= lookback_start) & (df.index <= current_period)]
        
        assets = df_lookback['Ticker'].unique()
        residuals_list = []
        for ticker in tqdm(assets):
            df_ticker = df_lookback[df_lookback['Ticker'] == ticker]
            df_ticker = df_ticker[['return', 'ibov_returns']].dropna() 
            if len(df_ticker) == 0:
                continue
            
            # Residuals
            X = sm.add_constant(df_ticker['ibov_returns'])
            y = df_ticker['return']
            model = sm.OLS(y, X).fit()
            resid = model.resid
            residuals_list.append(pd.DataFrame({ticker: resid}))
            
        if residuals_list:
            residuals = pd.concat(residuals_list, axis=1)
        
        # Deal with NaNs
        min_non_na = int(0.9 * len(residuals))
        residuals.dropna(thresh=min_non_na, axis=1, inplace=True)
        residuals = residuals.ffill().bfill()
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(residuals.T)
        
        # DBSCAN
        dbscan = DBSCAN(eps=db_eps, min_samples=db_min_samples)
        cluster_labels = dbscan.fit_predict(pca_result)
        
        # Dist
        long_positions = []
        short_positions = []
        
        # Cls
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]  # Excluindo ruído
        
        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            
            cluster_center = pca_result[cluster_indices].mean(axis=0)
            distances_to_center = np.linalg.norm(pca_result[cluster_indices] - cluster_center, axis=1)
            
            # Get assets
            sorted_indices = cluster_indices[np.argsort(distances_to_center)]
            
            n_long = min(8, len(sorted_indices))  
            n_short = min(7, len(sorted_indices))  
            
            for i in range(n_long):
                long_positions.append(residuals.columns[sorted_indices[i]])
            for i in range(n_short):
                short_positions.append(residuals.columns[sorted_indices[-(i+1)]])
        
        long_dict[current_period] = long_positions 
        short_dict[current_period] = short_positions 
        
        if first_iteration:
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', marker='o')
            plt.title(f"DBSCAN Clustering (eps={db_eps}, min_samples={db_min_samples}) - {current_period.date()}")
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.colorbar(label='Cluster Label')
            first_iteration = False
            
            # Id pos
            for i, ticker in enumerate(residuals.columns):
                if ticker in long_positions:
                    plt.annotate(ticker, (pca_result[i, 0], pca_result[i, 1]), color='blue', fontsize=8, fontweight='bold')
                elif ticker in short_positions:
                    plt.annotate(ticker, (pca_result[i, 0], pca_result[i, 1]), color='red', fontsize=8, fontweight='bold')
            
            plt.show()
        
        
        if turn_period == 'monthly':
            current_period += pd.offsets.MonthBegin(1)
        elif turn_period == 'weekly':
            current_period += pd.offsets.Week(1)
        else:
            print('error...')

    return long_dict, short_dict