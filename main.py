import tempfile
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

from data_handler import *
from data_utils import *
from strategies import * 
from backtesting import *



# COMMAND
weekly = False
liq_petr = 0.001
long_biased_opt = False
start_period = pd.to_datetime('2011-01-01')
test_period = pd.to_datetime('2018-12-01')
end_bt = '2019-01-31'
use_stop = False

# Get data
tmp_path = tempfile.mkdtemp()
t = Tickers_B3(tmp_path)
t.open_file()
assets, prices_enfoque, sectors_b3 = t.get_tickers()
sectors_b3['sector'].unique()

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
# df['return'] = df.groupby('Ticker')['Open'].transform(lambda x: np.log(x).diff())
# ibov['ibov_returns'] = np.log(ibov['Adj Close']/ibov['Adj Close'].shift(1))
df['return'] = df.groupby('Ticker')['Open'].transform(lambda x: x.pct_change())
ibov['ibov_returns'] = ibov['Adj Close'].pct_change()
df = pd.merge(df, ibov['ibov_returns'], on='Date')

# Test period (only one run): pre-backtesting
df_all_data = df.copy()
df = df[(df.index >= start_period) & (df.index <= test_period)]

# Liquidity filter
stock_reference = 'PETR4'
liquidity_filter = pd.DataFrame({'Vol_filter': df[df.Ticker==stock_reference]['Financial_Volume'].rolling(21).mean()})

# Apply liquidity: this is just to avoid stocks with no liquidity at all
df['rolling_mean'] = df.groupby('Ticker')['Financial_Volume'].transform(lambda x: x.rolling(window=21).mean())
df = pd.merge(df.reset_index(), liquidity_filter, on=['Date'], how='inner')

# obs.: teste inicial com 0.05 e 3 ativos comprados e vendidos
liquidity_mask = (df['rolling_mean'] < df['Vol_filter'] * liq_petr) # & (df['Close'] > 1.00) & (df['Close'] < 150)) # careful not to change df size 
date_backup = df['Date'].copy()
ticker_backup = df['Ticker'].copy()

df.loc[liquidity_mask, :] = np.nan
df['Date'] = date_backup
df['Ticker'] = ticker_backup

if weekly:
    period_port = 'weekly'
else:
    period_port = 'monthly'

# benchmark strategy
long_dict, short_dict = bench_ll(df, frequency=period_port)

# main strategy
# long_dict, short_dict = main_strategy_dbscan(df, start_period, lookback_period=30, db_eps=0.8, db_min_samples=4, turn_period=period_port) # 90
# long_dict, short_dict = main_strategy_kmeans(df, start_period, lookback_period=30)
# long_dict, short_dict = main_strategy_dbscan_sectorized(df, start_period, lookback_period=30, 
#                                                         db_eps=0.8, db_min_samples=4, turn_period='monthly')

# backtesting 
print('Running backtest...')
print('Dados para gráfico:')
input_data_plot = input()

fee = 0.005 * 0.10
equity = 100_000
round_control = 0
cdi_efficiency = 0.8
log_operations = []
buy_control, short_control, pos_control, bt_control = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
cash_flow, cash_flow_control = pd.DataFrame(), pd.DataFrame()
positions = False

# CDI
codigo_bcb = 11
bacen_api = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
selic = pd.read_json(bacen_api)
selic['data'] = pd.to_datetime(selic[ 'data'], dayfirst = True)
selic = selic.rename(columns={'data': 'Data', 'valor': 'Selic'})
selic = selic.set_index('Data')
selic = selic['Selic'] / 100


for date, _ in long_dict.items():
    
    print(f'Equity {equity} in {date}')
    
    date_ = find_last_day_of_month(date, ibov.index)
    date_bt = date_ + pd.offsets.MonthBegin(1)    
    
    if positions:
        
        # Valorizacao da posicao em caixa
        cdi_date_in = cash_flow_control.loc[round_control-1, 'data']
        cdi_date_out = date_
        cash = cash_flow_control.loc[round_control-1, 'cash']
        
        # print(date)
        # print(f'Cash: {cash}')
        
        handle_cdi = selic.copy()
        handle_cdi = handle_cdi[(handle_cdi.index>=cdi_date_in) & (handle_cdi.index<=cdi_date_out)]
        cum_cdi = handle_cdi.add(1).cumprod()-1
        
        # PENDENTE -----------------------
        cash_applied_cdi = cash + (cash * (cum_cdi.iloc[-1] * cdi_efficiency))
        cash_flow_control.loc[round_control-1, 'cash'] = cash_applied_cdi
        
        if len(buy_control) == 0 or len(short_control) == 0:
            print('Error: prob empty df...')
            break
        
        handle_positions, equity = handling_positions(pos_control, cash_flow_control, prices_enfoque, 
                                                      date_, fee, round_control, cum_cdi, stop=use_stop)
        bt_control = pd.concat([bt_control, handle_positions], axis=0)
        log_operations.append(handle_positions)
    
    if equity < 5000:
        print(f'Equity too low in {date}')
        break
    
    buy_port = pd.DataFrame({'assets': long_dict[date]})
    short_port = pd.DataFrame({'assets': short_dict[date]})
    if len(buy_port) == 0:
        continue
    
    buy_control, total_equity_usage_buy, cash_buy = make_positions(buy_port, prices_enfoque, equity, date_bt, fee, 
                                                                   round_control, buy=True, atr_values=None, long_biased=long_biased_opt)
    short_control, total_equity_usage_short, cash_short = make_positions(short_port, prices_enfoque, equity, date_bt, fee, 
                                                                         round_control, buy=False, atr_values=None, long_biased=long_biased_opt)
    
    # if len(buy_control) > 0:
    #     print('Posicao total comprada:')
    #     print(np.sum(buy_control.fin))
    # if len(short_control) > 0:
    #     print('Posicao total vendida:')
    #     print(np.sum(short_control.fin))
    
    pos_control = pd.concat([buy_control, short_control])
    cash_flow_control = handle_cash_flow(cash_flow, date_, equity, total_equity_usage_buy, total_equity_usage_short, cash_buy, cash_short, round_control)
    
    round_control+=1
    positions = True
    
    # Desmontar operacoes finais
    if date == list(long_dict.keys())[-1]:
        handle_positions, equity = handling_positions(pos_control, cash_flow_control, prices_enfoque, 
                                                      date_, fee, round_control, cum_cdi, stop=use_stop)
        
        cash_flow_control.loc[round_control, 'initial_equity'] = equity
        cash_flow_control.loc[round_control, 'data'] = end_bt
        
        bt_control = pd.concat([bt_control, handle_positions], axis=0)
        log_operations.append(handle_positions)


if weekly:
    bt_period_choice = 52
else:
    bt_period_choice = 12
    
operations_log = pd.concat(log_operations, axis=0)
# operations_log.to_excel(Path(Path.home(), 'Desktop', 'db3.xlsx'))
# cash_flow.to_excel(Path(Path.home(), 'Desktop', 'cash_flow_ll.xlsx'))

# Comissions
total_comission = np.sum(operations_log.comission)
print(f'Taxas pagas: {total_comission}')

# Results
bt_result = cash_flow_control[['data', 'initial_equity']]
bt_result.set_index('data', inplace=True)
bt_result.index.name = 'Date'
bt_result.index = pd.to_datetime(bt_result.index)
bt_result = bt_result.iloc[:-1,:]

# Merge str  ibov
btest = pd.merge_asof(bt_result, ibov['Adj Close'], on='Date', direction='backward')
btest = btest.rename(columns={'Adj Close': 'ibov', 'initial_equity':'final_return'})
btest.set_index('Date', inplace=True)
btest = btest.pct_change()
btest.dropna(inplace=True)

# Metrics
bt_result = metrics(btest, rf=0, period_param=bt_period_choice)
results_t = pd.DataFrame.from_dict(bt_result, orient='index', columns=['Métricas'])
print(results_t)
# results.to_excel(Path(Path.home(), 'Desktop', 'results.xlsx'))

# Ibov metrics
vol_ibov = btest.ibov.std() * np.sqrt(bt_period_choice)
n = len(btest.ibov)/bt_period_choice
total_return = btest.ibov.add(1).cumprod()
cagr_ibov = total_return.iloc[-1]**(1/n) - 1
sharpe_ibov = cagr_ibov / vol_ibov

# Backtest stats
def add0(df):
    df.loc[-1] = 0
    df.index = df.index + 1
    df = df.sort_index()
    return df

# insert the new row at the beginning of the datafra
btest = add0(btest.reset_index())
# btest.Data.iloc[0] = btest.Data.iloc[1] - dt.timedelta(days=1)
btest.loc[btest.index[0], 'Date'] = btest.loc[btest.index[1], 'Date'] - dt.timedelta(days=1)
btest.set_index('Date', inplace=True)

# Plot
if weekly:
    cdi_monthly = selic.resample('W-FRI').apply(lambda x: (x + 1).prod() - 1)
else:
    cdi_monthly = selic.resample('M').apply(lambda x: (x + 1).prod() - 1)
cdi_reindexed = cdi_monthly.reindex(btest.index, method='ffill')
cdi_plot = cdi_reindexed.add(1).cumprod() - 1
plot_performance(btest.final_return, btest.ibov, results_t, cdi_plot, input_data_plot)


# metrics 
# start = datetime.strftime(btest.index[0], format='%Y-%m-%d')
# end = datetime.strftime(btest.index[-1], format='%Y-%m-%d')
# key = 'final_return'
# portfolio = None
# name_strategy = 'auroraSt1_tests_1924_s'
# get_report(
#             btest.final_return, btest.ibov, start, end, key, 0, portfolio, 
#             output_name=name_strategy)




