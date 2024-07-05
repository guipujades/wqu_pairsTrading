import tempfile
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm 
import statsmodels.api as sm


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

# Get residuals
n = 90
residuals = []
dates = []
names = []
for name, group in tqdm(df.groupby('sector')):
    
    sector_dates = []
    unique_dates = np.sort(group.index.unique())
    
    print(f'\nRunning {name}...')
    for i in tqdm(range(0, len(unique_dates) - n, n)):
        
        # Select df using dates
        date_range = unique_dates[i: i + n]
        chunk = group[group.index.isin(date_range)]
        
        # Feature matrix
        features = chunk.pivot_table(index='Date', columns='Ticker', values='return')
        X = features.dropna(thresh=int(0.85*len(date_range)), axis=1).fillna(method='ffill').fillna(method='bfill')
        X = X.reset_index().drop_duplicates(subset=['Date'], keep='first')
        X.set_index('Date', inplace=True)
        X = X.sort_index()
        
        # Target preprocessing
        target = chunk.reset_index().drop_duplicates(subset='Date', keep='first')[['Date', 'ibov_returns']]
        y = target.set_index('Date')
        y = y.sort_index()

        X, y = X.align(y, join='inner', axis=0)
        
        for stock in X.columns:
            X_Stock = X[stock]
            X_Stock = sm.add_constant(X_Stock)
            model = sm.OLS(y, X_Stock).fit()
            resid = model.resid
            
            residuals.append(resid)
            dates.extend(resid.index)
            names.extend([stock] * len(resid))
            
        
       




assets = df['Ticker'].unique()
residuals = pd.DataFrame(index=df.index)
for ticker in tqdm(assets):
    df_ = df[df['Ticker'] == ticker]
    df_ = df_[['return', 'ibov_returns']].dropna() 
    if len(df_) == 0:
        continue

    X = sm.add_constant(df_['ibov_returns'])
    y = df_['return']
    model = sm.OLS(y, X).fit()
    resid = model.resid

    
    
    
    
    
