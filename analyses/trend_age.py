import time
import keys
from binance.client import Client
import binance.exceptions as bx
import binance_funcs as funcs
import pandas as pd
from pprint import pprint
import sessions
from config import not_pairs
import matplotlib.pyplot as plt
import numpy as np

start = time.perf_counter()

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)
client = Client(keys.bPkey, keys.bSkey)

# Define Functions -----------------------------------------------------------------------------------------------------

def plot_results():
    x = np.arange(len(results.index))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, results['long_prob'], width, label='long')
    ax.bar(x + width / 2, results['short_prob'], width, label='short')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('relative probability gain')
    ax.set_title('periods to ignore')
    ax.legend()

    fig.tight_layout()
    plt.show()

def prepare_doublest(pair, arg_1, arg_2):
    session.indicators = {'ema-200', f"st-10-{arg_1}", f"st-10-{arg_2}"}
    try:
        df = funcs.prepare_ohlc(session, pair)
    except EOFError as e:
        print(pair, e)
        return
    session.compute_indicators(df)

    drop_cols = ['open', 'high', 'low', 'volume', f'st-10-{arg_1}-up', f'st-10-{arg_1}-dn', f'st-10-{arg_2}-up', f'st-10-{arg_2}-dn']
    df.drop(columns=drop_cols, inplace=True)

    df['bullish_ema'] = df.close > df['ema-200']
    df['bearish_ema'] = df.close < df['ema-200']
    df['bullish_loose'] = df.close > df[f'st-10-{float(arg_1)}']
    df['bearish_loose'] = df.close < df[f'st-10-{float(arg_1)}']
    df['bullish_tight'] = df.close > df[f'st-10-{float(arg_2)}']
    df['bearish_tight'] = df.close < df[f'st-10-{float(arg_2)}']

    df['in_long'] = df.loc[:, ['bullish_tight', 'bullish_loose', 'bullish_ema']].all(axis='columns')
    df['in_short'] = df.loc[:, ['bearish_tight', 'bearish_loose', 'bearish_ema']].all(axis='columns')

    drop_cols = ['bullish_ema', 'bearish_ema', 'bullish_loose', 'bearish_loose', 'bullish_tight', 'bearish_tight',
                 f'st-10-{arg_1}', 'ema-200']
    df.drop(columns=drop_cols, inplace=True)

    df['long_count'] = (
        df
        .in_long
        .groupby(df.in_long.diff().cumsum())
        .cumcount()
    )
    df['short_count'] = (
        df
        .in_short
        .groupby(df.in_short.diff().cumsum())
        .cumcount()
    )

    # print(df.head())

    long_ls.extend(list(df.loc[df.in_long.diff().shift(-1, fill_value=False), 'long_count']))
    short_ls.extend(list(df.loc[df.in_short.diff().shift(-1, fill_value=False), 'short_count']))

# session.indicators.update(['ema-200', f"ema-{ema_1}", f"ema-{ema_2}", f"atr-10-{atr}"])
# session.indicators.update(['hma-200', f"ema-{e,a_1}", f"ema-{ema_2}", f"atr-10-{atr}"])

# Declare Variables ----------------------------------------------------------------------------------------------------

st_1 = [3.0, 4.0, 5.0]
st_2 = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ema_1 = 12
ema_2 = 21
atr = 1.2
timeframe = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

cross_pairs = funcs.get_pairs('USDT', 'CROSS')
pairs = [pair for pair in cross_pairs if pair not in not_pairs]

# Run Loops ------------------------------------------------------------------------------------------------------------
for tf in timeframe:
    session = sessions.MARGIN_SESSION(tf, 0, 0.0002)
    session.max_length = 70080  # default is 201 for live trading, need much more for backtesting

    for x in st_1:
        for y in st_2:
            long_ls = []
            short_ls = []

            # compile lists of all trends that appear in the data, represented by their lengths
            for pair in pairs:
                prepare_doublest(pair, x, y)

            print(f"{tf} dst {x} {y} Total longs count: {len(long_ls)}, Total shorts count: {len(short_ls)}")

            # cap all lengths to 50 for simplicity
            long_capped = [min(l, 50) for l in long_ls]
            short_capped = [min(s, 50) for s in short_ls]

            # convert to pandas series
            long = pd.Series(data=long_capped, name='long')
            short = pd.Series(data=short_capped, name='short')

            # transform all those individual lengths into counts of each length
            long = long.groupby(long).count()
            short = short.groupby(short).count()

            # convert to dataframe
            results = pd.DataFrame({'long_count': long, 'short_count': short})

            # to produce a new column whos value in each row which is the sum of that row and all subsequent rows,
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

            long_res = [round(res, 3) for res in list(results['long_z_score'])]
            short_res = [round(res, 3) for res in list(results['short_z_score'])]
            print(f"long z-scores: {long_res[:20]}")
            print(f"shrt z-scores: {short_res[:20]}")

            # print(results.head())

            lap = time.perf_counter()
            elapsed = round(lap - start)
            print(f"{tf} dst {x} {y} completed, time taken so far: {elapsed // 60}m {elapsed % 60}s")
            print('')

end = time.perf_counter()
elapsed = round(end - start)
print(f"Time taken: {elapsed // 60}m {elapsed % 60}s")