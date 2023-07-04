import time
from functions import keys, binance_funcs as funcs
from binance.client import Client
import pandas as pd
import sessions
from config import not_pairs
import json
from pathlib import Path
import itertools
import statistics as stats
import _pickle

live = True
start = time.perf_counter()

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)
client = Client(keys.bPkey, keys.bSkey)

# Define Functions -----------------------------------------------------------------------------------------------------

# def plot_results():
#     x = np.arange(len(results.index))  # the label locations
#     width = 0.35  # the width of the bars
#
#     fig, ax = plt.subplots()
#     ax.bar(x - width / 2, results['long_prob'], width, label='long')
#     ax.bar(x + width / 2, results['short_prob'], width, label='short')
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('relative probability gain')
#     ax.set_title('periods to ignore')
#     ax.legend()
#
#     fig.tight_layout()
#     plt.show()

def load_data(pair, limit):
    df = pd.read_pickle(f'/mnt/pi_2/ohlc_binance_15m/{pair}.pkl')
    df = funcs.resample_ohlc(session, df)
    df = df.tail(limit)
    df = df.reset_index(drop=True)

    return df


def prepare_doublest(pair, ma, mult_1, mult_2, limit):
    df = load_data(pair, limit)
    session.indicators = {f'{ma}-200', f"st-10-{mult_1}", f"st-10-{mult_2}"}
    df = session.compute_indicators(df)

    drop_cols = ['open', 'high', 'low', 'volume', f'st-10-{mult_1}-up', f'st-10-{mult_1}-dn', f'st-10-{mult_2}-up', f'st-10-{mult_2}-dn']
    df = df.drop(columns=drop_cols)

    df['bullish_ema'] = df.close > df[f'{ma}-200']
    df['bearish_ema'] = df.close < df[f'{ma}-200']
    df['bullish_loose'] = df.close > df[f'st-10-{float(mult_1)}']
    df['bearish_loose'] = df.close < df[f'st-10-{float(mult_1)}']
    df['bullish_tight'] = df.close > df[f'st-10-{float(mult_2)}']
    df['bearish_tight'] = df.close < df[f'st-10-{float(mult_2)}']

    df['first_long'] = df.loc[:, ['bullish_tight', 'bullish_loose', 'bullish_ema']].all(axis='columns')
    df['second_long'] = df.first_long & df.first_long.shift(1)
    df['first_short'] = df.loc[:, ['bearish_tight', 'bearish_loose', 'bearish_ema']].all(axis='columns')
    df['second_short'] = df.first_short & df.first_short.shift(1)

    drop_cols = ['bullish_ema', 'bearish_ema', 'bullish_loose', 'bearish_loose', 'bullish_tight', 'bearish_tight',
                 f'st-10-{mult_1}', f'{ma}-200']
    df = df.drop(columns=drop_cols)

    return df


def prepare_emax(pair, ma_type, atr, limit):
    df = load_data(pair, limit)
    session.indicators = {f'{ma_type}-200', "ema-12", "ema-21", f"atr-10-{atr}"}
    df = session.compute_indicators(df)

    df['bullish_bias'] = df.close > df[f'{ma_type}-200']
    df['bearish_bias'] = df.close < df[f'{ma_type}-200']
    df['bullish_emas'] = df['ema-12'] > df['ema-21']
    df['bearish_emas'] = df['ema-12'] < df['ema-21']
    df['low_above_atr'] = df.low > df[f"atr-10-{atr}-lower"]
    df['high_below_atr'] = df.high < df[f"atr-10-{atr}-upper"]

    df['first_long'] = df.loc[:, ['bullish_bias', 'bullish_emas', 'low_above_atr']].all(axis='columns')
    df['second_long'] = df.first_long & df.first_long.shift(1)
    df['first_short'] = df.loc[:, ['bearish_bias', 'bearish_emas', 'high_below_atr']].all(axis='columns')
    df['second_short'] = df.first_short & df.first_short.shift(1)

    drop_cols = ['bullish_emas', 'bearish_emas', 'bullish_bias', 'bearish_bias', 'low_above_atr',
                 'high_below_atr', f'{ma_type}-200']
    df = df.drop(columns=drop_cols)

    return df


def count_lengths(df, entry):
    df['long_count'] = (
        df[f'{entry}_long']
        .groupby(df[f'{entry}_long'].diff().cumsum())
        .cumcount()
    )
    df['short_count'] = (
        df[f'{entry}_short']
        .groupby(df[f'{entry}_short'].diff().cumsum())
        .cumcount()
    )

    long_ls.extend(list(df.loc[(df[f'{entry}_long'].diff().shift(-1, fill_value=False)) & (df[f'{entry}_long'] == True), 'long_count']))
    short_ls.extend(list(df.loc[(df[f'{entry}_short'].diff().shift(-1, fill_value=False)) & (df[f'{entry}_short'] == True), 'short_count']))


def count_pnls(df, inval_col_up, inval_col_dn, entry):
    df_l = df.loc[df[f'{entry}_long'].diff() == True, ['close', inval_col_up, f'{entry}_long']].copy()
    df_l['r'] = ((df_l.close - df_l[inval_col_up]) / df_l.close).shift(1)
    df_l['trades'] = df_l.close.loc[df_l[f'{entry}_long']==True].combine_first(df_l[inval_col_up])
    df_l['pnl_r'] = (df_l.trades.pct_change() - 0.0015) / df_l.r # 0.0015 is binance fees for round-trip trade

    if len(df_l):
        df_l = df_l.reset_index(drop=True)
        df_l = df_l.drop(index=0)
        long_rs.extend(list(df_l.pnl_r.loc[df[f'{entry}_long'] == False]))

    df_s = df.loc[df[f'{entry}_short'].diff() == True, ['close', inval_col_dn, f'{entry}_short']].copy()
    df_s['trades'] = df_s.close.loc[df_s[f'{entry}_short'] == True].combine_first(df_s[inval_col_dn])
    df_s['r'] = ((df_s[inval_col_dn] - df_s.close) / df_s.close).shift(1)
    df_s['pnl_r'] = ((df_s.trades.pct_change() * -1) - 0.0015) / df_s.r  # 0.0015 is binance fees for round-trip trade

    if len(df_s):
        df_s = df_s.reset_index(drop=True)
        df_s = df_s.drop(index=0)
        short_rs.extend(list(df_s.pnl_r.loc[df[f'{entry}_short'] == False]))


def calc_stats(long_ls, short_ls, long_rs, short_rs):
    len_l = len(long_ls)
    len_s = len(short_ls)
    median_l = stats.median(long_rs)
    median_s = stats.median(short_rs)
    print(f"Total longs count: {len_l}, Total shorts count: {len_s}")
    print(f"Avg long pnl (R): {median_l:.1f}, Avg short pnl (R): {median_s:.1f}"
          f", Combined: {(median_l / len_l)+(median_s / len_s)*(len_l + len_s):.1f}")


def create_length_counts_df(long_ls, short_ls, args):
    # cap all lengths to 50 for simplicity
    long_capped = [min(l, 50) for l in long_ls]
    short_capped = [min(s, 50) for s in short_ls]

    # convert to pandas series
    long = pd.Series(data=long_capped, name='long', dtype='float64')
    short = pd.Series(data=short_capped, name='short', dtype='float64')

    # transform all those individual lengths into counts of each length
    long = long.groupby(long).count()
    short = short.groupby(short).count()

    # convert to dataframe
    results = pd.DataFrame({'long_count': long, 'short_count': short})

    # to produce a new column whose value in each row which is the sum of that row and all subsequent rows,
    # first reverse the order
    # then use an expanding window sum, which returns the sum of every row up to the current row
    # then simply reverse the row order back to its original state
    results = results.sort_index(ascending=False)
    results['long_exp_sum'] = results.long_count.expanding().sum()
    results['short_exp_sum'] = results.short_count.expanding().sum()
    results = results.sort_index(ascending=True)

    # dividing the count at the current length by the total count of all lengths >= the current length gives the
    # probability that a trend will end at the current length
    results['long_prob'] = results.long_count / results.long_exp_sum
    results['short_prob'] = results.short_count / results.short_exp_sum

    # calculate z-scores of probabilities to allow generalised comparison
    results['long_z_score'] = (results['long_prob'] - results['long_prob'].mean()) / results['long_prob'].std()
    results['short_z_score'] = (results['short_prob'] - results['short_prob'].mean()) / results['short_prob'].std()

    res_name = '_'.join([str(arg) for arg in args])
    all_results[res_name] = {
        'long': list(results['long_z_score']),
        'short': list(results['short_z_score'])
    }

    long_res = [round(res, 3) for res in list(results['long_z_score'])]
    short_res = [round(res, 3) for res in list(results['short_z_score'])]

    return long_res, short_res


# Declare Variables ----------------------------------------------------------------------------------------------------

limit = 1000 # limit the historical lookback period for the tests
strats = {'dst': prepare_doublest, 'emax': prepare_emax}


# timeframes = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
timeframes = ['15m', '30m', '1h', '2h', '4h']

ma_types = ['ema', 'hma']
# ma_types = ['ema']

entries = ['first', 'second'] # enter on the first close after a signal is generated or the second
# entries = ['second']

st_1 = [3.0, 4.0, 5.5, 7.5, 10.0]
# st_1 = [3.0]
st_2 = [1.0, 1.2, 1.5, 1.9, 2.4, 2.9]
# st_2 = [1.0]

ema_1 = 12
ema_2 = 21
atrs = [1.0, 1.2, 1.5, 1.9, 2.4, 3.0]
# atrs = [1.0]

cross_pairs = funcs.get_pairs('USDT', 'CROSS')
pairs = [pair for pair in cross_pairs if pair not in not_pairs]
# pairs = ['BTCUSDT']

all_results = {}
pnls_df = pd.DataFrame()

print(f"\n{'-' * 100}\n")

# Run Loops ------------------------------------------------------------------------------------------------------------
strategy = 'emax'
for tf, ma_type, entry in itertools.product(timeframes, ma_types, entries):
    print(f"Running {strategy}{ma_type} {tf} {entry} entry Tests")
    session = sessions.MARGIN_SESSION(tf, 0, 0.0002)
    session.max_length = 70080  # default is 201 for live trading, need much more for backtesting

    for atr in atrs:
        long_ls = []
        long_rs = []
        short_ls = []
        short_rs = []

        # compile lists of all trends that appear in the data, represented by their lengths and pnls
        for pair in pairs:
            df = strats[strategy](pair, ma_type, atr, limit)
            count_lengths(df, entry)
            count_pnls(df, f'atr-10-{atr}-lower', f'atr-10-{atr}-upper', entry)

        calc_stats(long_ls, short_ls, long_rs, short_rs)

        args = (strategy, tf, ma_type, entry, atr)
        long_res, short_res = create_length_counts_df(long_ls, short_ls, args)

        lap = time.perf_counter()
        elapsed = round(lap - start)
        print(f"{atr} completed, time taken so far: {elapsed // 60}m {elapsed % 60}s")
        print('')

    if live:
        filepath = Path(f'/home/ross/Documents/backtester_2021/analyses/{strategy}_{ma_type}_trend_age.json')
    else:
        filepath = Path(f'/home/ross/Documents/backtester_2021/analyses/test_{strategy}_{ma_type}_trend_age.json')
    with open(filepath, 'w') as file:
        json.dump(all_results, file)

    print(f"\n{'-' * 100}\n")

strategy = 'dst'
for tf, ma_type, entry in itertools.product(timeframes, ma_types, entries):
    print(f"Running {strategy} {tf} {ma_type} {entry} entry Tests")
    session = sessions.MARGIN_SESSION(tf, 0, 0.0002)
    session.max_length = 70080  # default is 201 for live trading, need much more for backtesting

    for x, y in itertools.product(st_1, st_2):
        long_ls = []
        long_rs = []
        short_ls = []
        short_rs = []

        # compile lists of all trends that appear in the data, represented by their lengths and pnls
        for pair in pairs:
            try:
                df = strats[strategy](pair, ma_type, x, y, limit)
                count_lengths(df, entry)
                count_pnls(df, f'st-10-{y}', f'st-10-{y}', entry)
            except EOFError as e:
                print(pair, e)
                continue
            except _pickle.UnpicklingError as e:
                print(pair, e)
                continue

        calc_stats(long_ls, short_ls, long_rs, short_rs)

        args = (strategy, tf, ma_type, entry, x, y)
        long_res, short_res = create_length_counts_df(long_ls, short_ls, args)

        lap = time.perf_counter()
        elapsed = round(lap - start)
        print(f"{x} {y} completed, time taken so far: {elapsed // 60}m {elapsed % 60}s")
        print('')

    if live:
        filepath = Path(f'/home/ross/Documents/backtester_2021/analyses/{strategy}_{ma_type}_trend_age.json')
    else:
        filepath = Path(f'/home/ross/Documents/backtester_2021/analyses/test_{strategy}_{ma_type}_trend_age.json')
    with open(filepath, 'w') as file:
        json.dump(all_results, file)

    print(f"\n{'-' * 100}\n")

end = time.perf_counter()
elapsed = round(end - start)
print(f"Time taken: {elapsed // 60}m {elapsed % 60}s")

# 2 strategies
# 5 timeframes
# 2 ma types
# 2 entries
# 36 param combinations
# all pairs
#
# 1440 total tests on all pairs * ~30s per test = 243 minutes total
