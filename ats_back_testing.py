import pandas as pd
from pathlib import Path
import json
from pprint import pprint
import indicators as ind
import binance_funcs as funcs
import datetime
import time
from itertools import product

print(f"{datetime.datetime.now().strftime('%d/%m/%y %H:%M')} - Running ATS_Z Backtesting")

all_start = time.perf_counter()

# TODO investigate relationship between ats_z lookback and outcome
# TODO investigate relationship between rr_ratio and outcome
# TODO investigate relationship between atr_mult and outcome
# TODO investigate relationship between doji score and outcome

# TODO try doing the same test with completely random entries and see how the outcome compares to ats_z filtered entries

def track_oco(trade: dict, df: pd.DataFrame) -> dict:
    exit_price = None
    exit_time = None
    for row in df.itertuples():
        if row.low < trade['inval']:
            exit_price = trade['inval']
            exit_time = row.timestamp.timestamp()
            break
        if row.high > trade['target']:
            exit_price = trade['target']
            exit_time = row.timestamp.timestamp()
            break

    if exit_price:
        return dict(
            pnl=(exit_price - trade['entry']) / trade['entry'],
            duration=exit_time - trade['time'],
            exit_price=exit_price,
            exit_time=row.timestamp.strftime('%d/%m/%y %H:%M')
        )
    else:
        return None


def track_trail(trade: dict) -> dict:
    tf_map = {'15m': '15T', '30m': '30T', '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H',
              '8h': '8H', '12h': '12H', '1d': '1D', '3d': '3D', '1w': '1W'}

    hours = int(trade['timeframe'][:-1])
    lookback = hours * 10 * 3_600  # calculating the extra seconds needed to calculate atr bands

    df = prepare_trade(trade, lookback)
    htf_df = df.resample(tf_map[trade['timeframe']], on='timestamp').agg({'open': 'first',
                                                                          'high': 'max',
                                                                          'low': 'min',
                                                                          'close': 'last',
                                                                          'base_vol': 'sum',
                                                                          'quote_vol': 'sum',
                                                                          'num_trades': 'sum',
                                                                          'taker_buy_base_vol': 'sum',
                                                                          'taker_buy_quote_vol': 'sum'})

    atr_len = 5
    htf_df = ind.atr_bands(htf_df, atr_len, trade['atr_mult'])

    atr_col = f"atr-{atr_len}-{trade['atr_mult']}-lower"
    lower = htf_df[atr_col].resample('1T').ffill()
    lower = lower.sort_index().reset_index(drop=True)
    new_df = df.head(len(lower)).reset_index(drop=True)
    new_df = pd.concat([new_df, lower], axis=1)

    cut_mins = int(lookback / 60)  # turning the seconds from above into minutes to cut from the dataframe
    if len(new_df) < cut_mins:
        return None
    new_df = new_df.drop(index=list(range(cut_mins)))
    new_df = new_df.reset_index(drop=True)

    new_df.at[0, atr_col] = trade['inval']  # make sure cummax for trail starts with the original invalidation value
    new_df['trail'] = new_df[atr_col].cummax()

    # print(new_df.resample(tf_map[trade['timeframe']], on='timestamp').agg({'low': 'min',
    #                                                                        'close': 'last',
    #                                                                        'num_trades': 'sum',
    #                                                                        atr_col: 'last',
    #                                                                        'trail': 'last'}))

    exit_price = None
    exit_time = None
    for row in new_df.itertuples():
        if row.low < row.trail:
            exit_price = row.trail
            exit_time = row.timestamp.timestamp()
            break

    if exit_price:
        return dict(
            pnl=(exit_price - trade['entry']) / trade['entry'],
            duration=exit_time - trade['time'],
            exit_price=exit_price,
            exit_time=row.timestamp.strftime('%d/%m/%y %H:%M')
        )
    else:
        return None


timeframes = ['1h', '4h', '6h', '8h', '12h']
pairs = funcs.get_pairs()#[::100]
tf = '1h'

min_z = 2
atsz_lb = 200
mults = [#1,
         2, 3#, 4
         ]
rr_ratios = [#1,
             2#, 3
             ]

count = 0
completed = 0
trade_log = dict(
    tf=[],
    pair=[],
    exit=[],
    atr_mult=[],
    start=[],
    entry_price=[],
    init_stop=[],
    exit_price=[],
    exit_time=[],
    avg_volume=[],
    vol_delta_1=[],
    vol_delta_10=[],
    vol_delta_100=[],
    stoch_rsi=[],
    ats_z=[],
    doji=[],
    engulfing=[],
    inside_bar=[],
    ema_200=[],
    ema_30_ratio=[],
    ema_200_ratio=[],
    rr_ratio=[],
    r=[],
    pnl_r=[],
    win=[],
    duration=[]
)
for n, pair in enumerate(pairs):
    split = time.perf_counter() - all_start
    completed = n/len(pairs)
    projected = round(split * ((1 - completed) / completed)) if n else 360000
    remaining_hours = f"{projected // 3600}h" if (projected // 3600) else ''
    remaining_mins = f"{(projected // 60) % 60}m"# if ((projected // 60) % 60) else ''
    remaining_secs = f"{projected % 60}s" if (projected % 60) else ''
    remaining = ' '.join([remaining_hours, remaining_mins, remaining_secs])
    print(f"{datetime.datetime.now().strftime('%d/%m/%y %H:%M')} - Backtesting, {completed:.1%} complete, {remaining} remaining")
    signals = []

    # get 1min data
    path_1m = Path(f"bin_ohlc_1m/{pair}.pkl")
    data_1m = pd.read_pickle(path_1m)
    data_1m = funcs.update_ohlc(pair, '1m', data_1m)
    data_1m = data_1m.tail(525600).reset_index(drop=True)

    # get 1h data
    path_1h = Path(f"bin_ohlc_1h/{pair}.pkl")
    data_1h = pd.read_pickle(path_1h)
    data_1h = funcs.update_ohlc(pair, '1h', data_1h)
    data_1h = data_1h.tail(8760).reset_index(drop=True)

    for mult, ratio in product(mults, rr_ratios):
        df = data_1h.copy()
        if tf != '1h':
            df = funcs.resample_ohlc(tf, 0, df)
        df['avg_volume'] = df.quote_vol.rolling(50).mean()
        df['vol_delta_1'] = (df.taker_buy_base_vol - df.base_vol) > 0
        df['vol_delta_10'] = (df.taker_buy_base_vol - df.base_vol).rolling(10).sum() > 0
        df['vol_delta_100'] = (df.taker_buy_base_vol - df.base_vol).rolling(100).sum() > 0
        df = ind.ats_z(df, atsz_lb)
        df['atsz_max'] = df.ats_z.rolling(10).max()
        df['ema_30'] = df.close.ewm(30).mean()
        df['ema_200'] = df.close.ewm(200).mean()
        df['bullish_ema'] = (df.ema_200 > df.ema_200.shift(5)).astype(int)
        df['stoch_rsi'] = ind.stoch_rsi(df.close, 14, 14)
        df = ind.atr_bands(df, 5, mult)
        df['atr_lower'] = df[f'atr-5-{mult}-lower']
        df = ind.inside_bars(df)
        df = ind.engulfing(df)
        df = ind.doji(df)
        df = ind.bull_bear_bar(df)
        df = df.dropna().reset_index(drop=True)

        for row in df.itertuples():
            atsz = row.atsz_max
            stoch_rsi = row.stoch_rsi
            # doji = 'doji' if row.bullish_doji else ''
            # engulf = 'engulf' if row.bullish_engulf else ''
            # ib = 'inside bar' if row.inside_bar else ''
            # candle = ' '.join([doji, engulf, ib])

            record_dict = {}
            record_dict['inval'] = row.atr_lower
            record_dict['inval_ratio'] = row.close / row.atr_lower # current price proportional to inval price

            # bullish_atsz = atsz > min_z
            # bullish_candles = row.inside_bar or row.bullish_doji or row.bullish_engulf

            if (record_dict['inval_ratio'] > 1):
                record_dict['target'] = row.close * (record_dict['inval_ratio'] ** ratio)
                record_dict['time'] = row.timestamp.timestamp()
                record_dict['pair'] = pair
                record_dict['timeframe'] = tf
                record_dict['rr_ratio'] = ratio
                record_dict['entry'] = row.close
                record_dict['avg_volume'] = row.avg_volume
                record_dict['vol_delta_1'] = row.vol_delta_1
                record_dict['vol_delta_10'] = row.vol_delta_10
                record_dict['vol_delta_100'] = row.vol_delta_100
                record_dict['stoch_rsi'] = stoch_rsi
                record_dict['ats_z'] = atsz
                record_dict['doji'] = row.bullish_doji
                record_dict['engulfing'] = row.bullish_engulf
                record_dict['inside_bar'] = row.inside_bar
                record_dict['min_z'] = min_z
                record_dict['atsz_lb'] = atsz_lb
                record_dict['atr_mult'] = mult
                record_dict['bullish_ema'] = row.bullish_ema
                record_dict['ema_30_ratio'] = row.close / row.ema_30
                record_dict['ema_200_ratio'] = row.close / row.ema_200

                signals.append(record_dict)

    signals = sorted(signals, key=lambda x: x['time'])

    ####################################################################################################################

    exit_type = 'oco'
    # if exit_type == 'trail': resample htf df to 1m and concat indicator columns

    for signal in signals:
        tf = signal.get('timeframe')
        count += 1
        trade_start = datetime.datetime.fromtimestamp(signal['time'])
        trade_ohlc = data_1m.loc[data_1m.timestamp > trade_start].reset_index(drop=True)
        if exit_type == 'oco':
            trade_result = track_oco(signal, trade_ohlc)
        elif exit_type == 'trail':
            trade_result = track_trail(signal, trade_ohlc)
        if not trade_result:
            continue

        pair = signal.get('pair')
        start_stamp = int(signal.get('time'))
        start = datetime.datetime.fromtimestamp(start_stamp)
        entry_price = signal.get('entry')

        init_stop = signal.get('inval')

        r = (entry_price - init_stop) / entry_price
        completed += 1
        pnl_pct = trade_result['pnl']
        pnl_r = trade_result['pnl'] / r
        duration = round(trade_result['duration'])
        trade_duration = f"{duration // 3600}h {(duration // 60) % 60}m {duration % 60}s"

        if trade_result['exit_price'] < signal['inval']:
            pprint(signal)
            print(f"{trade_result['exit_price'] = } {signal['inval'] = } {signal.get('inval') = }")

        # print(f"{start} {pair} {tf}, {signal['atr_mult']} {entry_price = }, {init_stop = }, "
        #       f"r value: {r:.2%}, {pnl_pct:.2%}, {pnl_r:.2f}R, {trade_duration}")

        trade_log['tf'].append(tf)
        trade_log['pair'].append(pair)
        trade_log['exit'].append(exit_type)
        trade_log['atr_mult'].append(signal['atr_mult'])
        trade_log['start'].append(start_stamp)
        trade_log['entry_price'].append(entry_price)
        trade_log['rr_ratio'].append(signal['rr_ratio'])
        trade_log['init_stop'].append(init_stop)
        trade_log['exit_price'].append(trade_result['exit_price'])
        trade_log['exit_time'].append(trade_result['exit_time'])
        trade_log['avg_volume'].append(signal['avg_volume'])
        trade_log['vol_delta_1'].append(signal['vol_delta_1'])
        trade_log['vol_delta_10'].append(signal['vol_delta_10'])
        trade_log['vol_delta_100'].append(signal['vol_delta_100'])
        trade_log['stoch_rsi'].append(signal['stoch_rsi'])
        trade_log['ats_z'].append(signal['ats_z'])
        trade_log['doji'].append(signal['doji'])
        trade_log['engulfing'].append(signal['engulfing'])
        trade_log['inside_bar'].append(signal['inside_bar'])
        trade_log['ema_200'].append(signal['bullish_ema'])
        trade_log['ema_30_ratio'].append(signal['ema_30_ratio'])
        trade_log['ema_200_ratio'].append(signal['ema_200_ratio'])
        trade_log['r'].append(r)
        trade_log['pnl_r'].append(pnl_r)
        trade_log['win'].append(pnl_r > 0)
        trade_log['duration'].append(trade_duration)

########################################################################################################################

results_df = pd.DataFrame.from_dict(trade_log)
results_df['start'] = pd.to_datetime(results_df.start*1000000000)

results_df = results_df.loc[results_df.exit == 'oco']
results_df.to_pickle(f'forest_input_{tf}.pkl')

# print('overall')
# print(f"{len(results_df)} trades, winrate: {len(results_df.loc[results_df.pnl_r > 0]) / len(results_df):.1%}, "
#           f"avg_pnl_r: {results_df.pnl_r.mean():.2f}\n")
#
# for i, group in results_df.groupby(['atr_mult']):
#     print(f"{i}, {len(group)} trades, winrate: {len(group.loc[group.pnl_r > 0]) / len(group):.1%}, "
#           f"avg_pnl_r: {group.pnl_r.mean():.2f}")
#     print(group.sort_values('pnl_r', ascending=False).head())
#     print(group.sort_values('pnl_r', ascending=False).tail())

all_end = time.perf_counter()
elapsed = round(all_end - all_start)
print(f"Time taken: {elapsed // 60}m {elapsed % 60}s")
