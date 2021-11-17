import pandas as pd
import numpy as np

def get_signals(df, buy_thresh, sell_thresh):
    '''during an uptrend as defined by 20ema being above 200ema and price 
    being above st line, set 'trade ready'. if rsi subsequently drops below and 
    then crosses back above x, trigger a buy.
    if rsi goes above y and then crosses back below OR if price closes below 
    st line, trigger a sell
    * only conditions that actually trigger a trade need to be defined as a 
    specific moment in time (eg a cross up or down), other conditions which set 
    the stage should be more diffuse (eg price is above supertrend line). if 
    there are several conditions which are only true on individual moments, 
    the chances of them lining up are much lower, ie good trades will be missed.
    im not looking for price crossing above the st line because that is not what 
    triggers a buy, i only care whether it is above or below, but a cross BELOW
    the st line will trigger a stop, so that condition must be a cross, not 
    just a simple less-than'''
    signals = [np.nan]
    s_buy = [np.nan]
    s_sell = [np.nan]
    s_stop = [np.nan]
    stop_price = 0
    trade_ready = 0
    in_pos = 0
    buys = 0
    sells = 0
    stops = 0
    for i in range(1, len(df)):
        ### trade conditions
        trend_up = (df.loc[i, '20ema'] > df.loc[i, '200ema'])
        st_up = (df.close[i] > df.st[i]) # not a trigger so doesnt need to be a cross
        cross_down = (df.close[i] < df.st[i]) and (df.close[i-1] >= df.st[i-1]) # this is a trigger so does need to be a cross
        rsi_buy = (df.rsi[i] >= buy_thresh) and (df.rsi[i-1] < buy_thresh)
        rsi_sell = (df.rsi[i] <= sell_thresh) and (df.rsi[i-1] > sell_thresh)
            
        if trend_up and st_up and in_pos == 0:
            trade_ready = 1
        else:
            trade_ready = 0
        
        if trade_ready == 1 and rsi_buy:
            sl = df.st[i] * 0.995
            signals.append(f'buy, init stop @ {sl}')
            stop_price = df.st[i]
            s_buy.append(df.close[i])
            s_sell.append(np.nan)
            s_stop.append(np.nan)
            in_pos = 1
            buys += 1
        elif in_pos and cross_down:
            signals.append('stop')
            s_buy.append(np.nan)
            s_sell.append(np.nan)
            s_stop.append(df.close[i-1])
            in_pos = 0
            stops += 1
        elif in_pos and rsi_sell:
            signals.append('sell')
            s_buy.append(np.nan)
            s_sell.append(df.close[i])
            s_stop.append(np.nan)
            in_pos = 0
            sells += 1
        else:
            signals.append(np.nan)
            s_buy.append(np.nan)
            s_sell.append(np.nan)
            s_stop.append(np.nan)

        if in_pos == 1 and (df.st[i] > stop_price):
            stop_price = df.st[i]
            
    
    
    df['signals'] = signals
    
    pos_list = [0, 0]
    for p in range(1, len(df.index)):
        if pd.isnull(df.at[p, 'signals']):
            pos_list.append(pos_list[-1])
        elif df.at[p, 'signals'][:3] == 'buy':
            pos_list.append(1)
        elif df.at[p, 'signals'] == 'sell':
            pos_list.append(0)
        elif df.at[p, 'signals'] == 'stop':
            pos_list.append(0)
        else:
            pos_list.append(pos_list[-1])
    
    df['in_pos'] = pos_list[:-1]
    
    exe_price = []
    for e in range(len(df.index)):
        if pd.isnull(df.at[e, 'signals']):
            exe_price.append(df.at[e, 'close'])
        elif df.at[e, 'signals'] == 'stop':
            exe_price.append(df.at[e-1, 'st'])
        else:
            exe_price.append(df.at[e, 'close'])
    df['exe_price'] = exe_price
    df['exe_roc'] = df['exe_price'].pct_change()
    df['roc'] = df['close'].pct_change()
    
    evo = [1]
    hodl_evo = [1]
    for e in range(1, len(df.index)):
        evo.append(evo[-1] * (1 + (df.at[e, 'exe_roc'] * df.at[e, 'in_pos'])))
        hodl_evo.append(hodl_evo[-1] * (1 + (df.at[e, 'roc'])))
    df['pnl_evo'] = evo
    df['hodl_evo'] = hodl_evo
    
    # print(df.drop(['open', 'high', 'low', 'volume', 'st_u', 'st_d', 
    #                '20ema', '200ema', 'rsi'], axis=1).tail())
        
    
    sb, sse, sst = pd.Series(s_buy), pd.Series(s_sell), pd.Series(s_stop)
    sb.index, sse.index, sst.index = df.index, df.index, df.index
    return buys, sells, stops, sb, sse, sst

