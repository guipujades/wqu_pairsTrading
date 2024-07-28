import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Backtest
def find_last_day_of_month(period, date_index):
    month_dates = date_index[(date_index.year == period.year) & (date_index.month == period.month)]
    if len(month_dates) > 0:
        return month_dates[-1]
    else:
        return None

def metrics(df, rf=0, period_param=252) -> dict:
    results = {}
    returns = df.final_return

    # retorno total
    total_return = returns.add(1).cumprod()
    t_return = total_return - 1
    results['Retorno'] = round(t_return.iloc[-1],3)
    
    # volatilidade
    vol = returns.std() * np.sqrt(period_param)
    results['Volatilidade'] = round(vol,3)

    # max_drawdown
    t_return_max = total_return.cummax()
    dd = t_return_max - total_return
    result_dd = (dd / t_return_max).max()
    results['Drawdown'] = round(result_dd,3)

    # cagr
    n = len(df)/period_param
    cagr = total_return.iloc[-1]**(1/n) - 1
    results['CAGR'] = round(cagr,3)

    # sharpe
    sharpe = cagr / vol
    results['Sharpe'] = round(sharpe,3)

    # sortino
    returns_neg = np.where(returns>0,0,returns)
    volatility_neg = pd.Series(returns_neg[returns_neg!=0]).std() * np.sqrt(period_param)
    sortino = (cagr - rf) / volatility_neg
    results['Sortino'] = round(sortino,3)

    # beta
    covariance = df['final_return'].cov(df['ibov'])
    variance = df['ibov'].var()
    beta = covariance / variance
    results['Beta'] = round(beta,3)

    # loss control
    loss_days = [i for i in returns if i<0]
    loss = len(loss_days) / len(returns)
    results['Dias_perda'] = round(loss,3)

    return results


def plot_performance(returns, ibov, metrics_plot, cdi):
    # Calcula o retorno acumulado para a estratégia, Ibovespa e CDI
    cum_return = returns.add(1).cumprod() - 1
    bench_cum_return = ibov.add(1).cumprod() - 1
    cdi_cum_return = cdi.add(1).cumprod() - 1
    cdi_cum_return = cdi.copy()

    metrics_text = str(metrics_plot)
    
    # Plota os retornos acumulados
    plt.figure(figsize=(15, 8))
    plt.plot(cum_return.index, cum_return, label='Long & Short Lead-Lag', color='darkblue')
    plt.plot(bench_cum_return.index, bench_cum_return, label='Ibovespa', color='gray', linestyle='--')
    # plt.plot(cdi_cum_return.index, cdi_cum_return, label='CDI', color='gray', linestyle='-.')
    
    # plt.text(0.7, 0.2, metrics_text, fontsize=12, transform=plt.gcf().transFigure)
    
    # Configurações do gráfico
    plt.title('Performance')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()


def handle_cash_flow(cash_flow, date, equity, total_equity_usage_buy, total_equity_usage_short, cash_buy, cash_short, round_control):
    
    total_equity_usage = total_equity_usage_buy + total_equity_usage_short
    cash = cash_buy + cash_short
    
    cash_flow.loc[round_control, 'data'] = date
    cash_flow.loc[round_control, 'initial_equity'] = equity
    cash_flow.loc[round_control, 'equity_usage'] = total_equity_usage
    cash_flow.loc[round_control, 'terminal_equity'] = equity - total_equity_usage
    cash_flow.loc[round_control, 'cash'] = cash
    if round_control == 0:
        cash_flow.index.name = 'round'
    
    return cash_flow


def make_positions(use_equity, prices_enfoque, equity, date, fee, round_control, buy=True, adjust_equity= 0.3, atr_values=None):
    
    if atr_values is not None:
        
        atr_values = atr_values[atr_values.index==date]
        use_equity = pd.merge(use_equity, atr_values, left_on='assets', right_on='Ticker', how='left')

        # total_atr = use_equity['ATR'].sum()
        # use_equity['weights'] = use_equity['ATR'] / total_atr
        
        # use_equity.drop('Ticker', axis=1, inplace=True)
        # # Essa parte aqui pode gerar desencaixe de ativos, entao nao e ideal
        # use_equity.fillna(0, inplace=True)
        
        use_equity['reciprocal_ATR'] = 1 / (use_equity['ATR'] + 1e-6)
        total_reciprocal_atr = use_equity['reciprocal_ATR'].sum()
        use_equity['weights'] = use_equity['reciprocal_ATR'] / total_reciprocal_atr
        use_equity.drop(['Ticker', 'reciprocal_ATR'], axis=1, inplace=True)
        use_equity.fillna(0, inplace=True)
    
    else:
        # If no ATR values provided, use equal weights
        use_equity['weights'] = 1 / len(use_equity)
    
    # use_equity['weights'] = 1 / len(use_equity)
    
    adjusted_equity = equity * adjust_equity # ajustar equity para evitar carteira alavancada
    use_equity['fin'] = use_equity['weights'] * adjusted_equity
    # assets = use_equity.assets

    av_dates = list(prices_enfoque.Date.unique())
    if date in av_dates:
        find_next_date = av_dates.index(date) + 1 # operar na data seguinte
        date = av_dates[find_next_date]
    
    if buy:
        
        if date in prices_enfoque.Date.unique():
            sec_df = prices_enfoque[prices_enfoque.Date==date]
            prices = sec_df[sec_df.Ticker.isin(list(use_equity.assets))][['Ticker', 'Open']]
            use_equity = pd.merge(use_equity, prices, left_on='assets', right_on='Ticker')
     
        else:
            prior_dates = ((prices_enfoque[pd.to_datetime(prices_enfoque.Date) < date]).Date).unique()
            if len(prior_dates) > 0:
               # Encontrar a data mais próxima anterior
               nearest_date = prior_dates.max()
               
               find_next_date = av_dates.index(nearest_date) + 1
               nearest_date = av_dates[find_next_date]
               
               sec_df = prices_enfoque[prices_enfoque.Date==nearest_date]
               prices = sec_df[sec_df.Ticker.isin(list(use_equity.assets))][['Ticker', 'Open']]
               
               counter = 0
               while len(prices) == 0 and counter < 3:
                   try:
                       find_next_date = av_dates.index(nearest_date) + 1
                       nearest_date = av_dates[find_next_date]
                       sec_df = prices_enfoque[prices_enfoque.Date==nearest_date]
                       prices = sec_df[sec_df.Ticker.isin(list(use_equity.assets))][['Ticker', 'Open']]
                       
                   except IndexError:  
                       break
                   counter += 1 
                       
               if len(prices) == 0:  
                   raise Exception(f'Não foi possível encontrar preço para {list(use_equity.assets)}')
                
               use_equity = pd.merge(use_equity, prices, left_on='assets', right_on='Ticker')
        
    else:
        
        if date in prices_enfoque.Date.unique():
            sec_df = prices_enfoque[prices_enfoque.Date==date]
            prices = sec_df[sec_df.Ticker.isin(list(use_equity.assets))][['Ticker', 'Open']]
            use_equity = pd.merge(use_equity, prices, left_on='assets', right_on='Ticker')
            use_equity['Open'] = use_equity['Open'] * -1
            
        else:
            prior_dates = ((prices_enfoque[pd.to_datetime(prices_enfoque.Date) < date]).Date).unique()
            if len(prior_dates) > 0:
                # Encontrar a data mais próxima anterior
                nearest_date = prior_dates.max()
               
                find_next_date = av_dates.index(nearest_date) + 1
                nearest_date = av_dates[find_next_date]
               
                sec_df = prices_enfoque[prices_enfoque.Date==nearest_date]
                prices = sec_df[sec_df.Ticker.isin(list(use_equity.assets))][['Ticker', 'Open']]
               
                counter = 0
                while len(prices) == 0 and counter < 3:
                    try:
                        find_next_date = av_dates.index(nearest_date) + 1
                        nearest_date = av_dates[find_next_date]
                        sec_df = prices_enfoque[prices_enfoque.Date==nearest_date]
                        prices = sec_df[sec_df.Ticker.isin(list(use_equity.assets))][['Ticker', 'Open']]
                       
                    except IndexError:  
                        break
                    counter += 1 
                       
                if len(prices) == 0:  
                    raise Exception(f'Não foi possível encontrar preço para {list(use_equity.assets)}')
               
                use_equity = pd.merge(use_equity, prices, left_on='assets', right_on='Ticker')
                use_equity['Open'] = use_equity['Open'] * -1
    
    
    use_equity.drop(columns=['Ticker'], inplace=True)
    use_equity = use_equity.rename(columns={'Open': 'price'})
    
    
    use_equity['volume'] = abs((use_equity['fin'] / use_equity['price']).astype(int))
    use_equity['volume'] = round(use_equity['volume'] / 100) * 100
    use_equity ['fin_volume'] = use_equity['price'] * use_equity['volume']
    use_equity['comission'] = abs(use_equity['fin_volume']) * fee
    use_equity['round'] = round_control
    
    if date not in prices_enfoque.Date.unique():
        date = pd.to_datetime(nearest_date)
        
    use_equity['date_in'] = date
    # use_equity.set_index('date', inplace=True)
    
    equity_usage = np.sum(use_equity['fin_volume'])
    total_comission = np.sum(use_equity['comission'])
    
    if buy:
        total_equity_usage = equity_usage + total_comission
        cash = equity - total_equity_usage
    else:
        total_equity_usage = 0
        cash = abs(equity_usage + total_comission)

    return use_equity, total_equity_usage, cash


def handling_positions(pos_control, cash_flow_control, prices_enfoque, date, fee, round_control, cum_cdi):
    
    
    cash_flow_control = cash_flow_control[cash_flow_control.index == round_control - 1]
    assets = list(pos_control.assets)
    
    if date in prices_enfoque.Date:
        sec_df = prices_enfoque[prices_enfoque.Date==date]
        prices = sec_df[sec_df.Ticker.isin(assets)][['Ticker', 'Open']]
        pos_control = pd.merge(pos_control, prices, left_on='assets', right_on='Ticker')
        
    else:
        prior_dates = (prices_enfoque[pd.to_datetime(prices_enfoque.Date) < date]).Date
        if not prior_dates.empty:
           # Encontrar a data mais próxima anterior
           nearest_date = prior_dates.max()
           sec_df = prices_enfoque[prices_enfoque.Date==nearest_date]
           prices = sec_df[sec_df.Ticker.isin(assets)][['Ticker', 'Open']]
           pos_control = pd.merge(pos_control, prices, left_on='assets', right_on='Ticker')

    pos_control['price_to_sell'] = np.where(pos_control['price'] > 0, pos_control['Open'], pos_control['Open']*-1)
    pos_control['fin_sell'] = pos_control['price_to_sell']  * pos_control['volume']
    pos_control['profit'] = pos_control['fin_sell'] - pos_control['fin_volume']
    
    # Taxas BTC
    short_comission_fees = abs(np.where(pos_control['fin_volume'] > 0, pos_control['fin_volume'] * fee, pos_control['fin_volume'] * 0.005))
    short_btc = np.where(pos_control['profit'] / pos_control['fin'] < 0, abs(pos_control['fin_volume'] * cum_cdi.iloc[-1]), abs(pos_control['profit'] / pos_control['fin']/10))
    
    pos_control['comission_out'] = short_comission_fees + short_btc
    
    
    pos_control['date_out'] = date
    
    # Redução do caixa em funcao das vendas desfeitas dos shorts
    equity_shorts = abs(np.sum(pos_control['fin_sell'][pos_control['fin_sell'] < 0]))
    new_cash_after_handling_shorts = list(cash_flow_control.cash)[0] - equity_shorts
    # Entrada em funcao das vendas das posicoes compradas
    equity_buy = abs(np.sum(pos_control['fin_sell'][pos_control['fin_sell'] > 0]))
    new_total_cash = equity_buy + new_cash_after_handling_shorts

    new_total_equity = new_total_cash - np.sum(pos_control['comission_out'])
    
    return pos_control, new_total_equity



