"""this is the continuous trader equivalent of setup_scanner, the central script that puts everything together and runs
it all"""

# from wootrade import Client as Client_w
from binance.client import Client as Client_b
import binance.enums as be
import mt.resources.keys as keys
from datetime import datetime, timedelta, timezone
import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics as stats
from pathlib import Path
from continuous import components
from time import perf_counter
from pprint import pprint
from itertools import permutations

all_start = perf_counter()

# tos = datetime.now().hour

pl.Config(tbl_cols=20, tbl_rows=50, tbl_width_chars=180)


# client = Client_w(keys.woo_key, keys.woo_secret, keys.woo_app_id, testnet=True)
# client = Client_b(keys.bPkey, keys.bSkey)

# TODO make a dashboard
# TODO correlation analysis to find pairs that don't move with the rest of the market
# TODO add drawdown to the pnl plots
# TODO work out how to apply the buffer concept to the backtests in the same way it applies to live trading, then remove
#  forecast quantisation and check how that changes things

# TODO import a list of all binance pairs i have data for, then test the pairs with the shortest history just to see if
#  it works. If so, start testing lots of different portfolios with flat portfolio weighting to see if i can do better
#  than i currently am with a manually selected portfolio

# TODO backtesting idea: to test which filters to use in portfolio construction, i could backtest jumbo portfolios with
#  different filters to see if there is a useful differentiation. eg if i want to test weekly rsi, i could make the
#  following jumbo portfolios and compare their backtests:
#  all pairs above 50 weekly rsi,
#  all pairs below 50 weekly rsi,
#  all pairs above 75 weekly rsi,
#  all pairs below 25 weekly rsi,
#  all pairs between 25 and 75 weekly rsi
#  this would need to be done with walk-forward testing, where i use one window to choose the coins, then test their
#  performance on the next window, then choose new coins and test them on the following window etc

# TODO i could make an oscillator-based strategy using the same technique i used with chanbreak, where i set up a
#  condition that looks at whether the oscilator is over or under a certain level eg 80% and 20%, and takes the diff of
#  those conditions, then uses pl.when to separate out just the moments when the oscillator crossed back inside those
#  extreme levels, indicating possible mean reversion. i could even get it to stay flat when the oscillator is at the
#  extremes to  really differentiate it from the trend-following strats, depending on whether that actually improves
#  things or not. i could test it with rsi, awesome osc, fisher transform etc

# TODO maybe i could engineer reversal indicators as temporary signals to reduce size. i don't know if this is a good
#  idea or not but instead of thinking of reversal/mean reversion signals in the same way as trend-following signals (ie
#  directional forecasts), i could instead think of them as a scalar from 1 down to 0 that shuts of all the directional
#  forecasts at any point where its likely they will be wrong (changes in trend) and then gradually eases of as the
#  trend-following forecasts gradually adjust to the new direction. I could also have something like adx or long-term
#  rsi doing a similar job to minimise attempted trend-following behaviour during chop.

# note: i think the forward feature selection is probably curve fitting, so i won't use it until i can forward test it
def sqntl_fwd(markets, n, weighting: str = 'weighted_lin'):
    """pick the best portfolio of n trading pairs from the list by adding one pair at a time and backtesting the
    combined sharpe ratio of each iteration"""

    best = markets[:1]
    remaining = markets[1:]
    high_score = 0

    while len(best) < n:
        results = []
        for r in remaining:
            interim = best.copy()
            interim.append(r)
            trader_2y = components.Trader(interim, 17520, 1)
            all_pnls = trader_2y.compare_backtests()
            stats = components.calc_perf(all_pnls[weighting])
            results.append({'trial': r, 'sharpe': stats['sharpe']})

        res_df = pl.DataFrame(results).sort('sharpe', descending=True)
        best_trial = res_df.item(0, 'trial')
        best_sharpe = res_df.item(0, 'sharpe')
        if best_sharpe > high_score:
            best.append(best_trial)
            remaining.remove(best_trial)
            high_score = best_sharpe
        else:
            break

    print(f"\nOptimum Portfolio: {best}")
    trader_2y = components.Trader(best, 17520, 1)
    all_pnls = trader_2y.compare_backtests()
    stats = components.calc_perf(all_pnls['weighted_lin'])
    components.print_stats('weighted_lin', stats)

    return best


def choose_by_length(minimum: int | str, maximum: int | str = 420000):
    """checks the length of ohlc history of each trading pair, then makes a list of all pairs whos history length falls
    within the stated range"""

    lengths = {'1 month': 8750, '2 months': 17500, '3 months': 26250, '6 months': 52500,
               '1 year': 105000, '2 years': 210000, '3 years': 315000, '4 years': 420000}

    if isinstance(minimum, str):
        minimum = lengths[minimum]
    if isinstance(maximum, str):
        maximum = lengths[maximum]

    info = {}
    data_path = Path("/home/ross/coding/modular_trader/bin_ohlc_5m")
    for pair_path in list(data_path.glob('*')):
        df = pl.read_parquet(pair_path)
        info[pair_path.stem] = len(df)

    return [p for p, v in info.items() if minimum < v <= maximum]


def backtest_all():
    # load each pair's ohlc one by one and compile a dict of statistics about them (backtested sharpe,ohlc length,
    # current daily/weekly volume, htf overbought/oversold etc) then put all those stats into a dataframe and filter out
    # the worst examples for each factor, then sort the rest by sharpe and pick the best 10/15/20. run this once a week
    # or so.
    pass


markets = [
    'BTCUSDT',
    'SOLUSDT',
    'PENDLEUSDT',
    'PHBUSDT',
    'MOVRUSDT',
    'ARKMUSDT',
    'JTOUSDT',
    'HIGHUSDT',
    'RNDRUSDT',
    'UMAUSDT',
    'PEOPLEUSDT',
    'TRBUSDT',
    'FTTUSDT',
    'FRONTUSDT',
    'KMDUSDT'
]

strategies = [
    'srsirev',
    # 'rsirev',
    'chanbreak',
    'ichitrend',
    # 'emaroc',
    'hmaroc'
]
# lookback window options: '4 years', '3 years', '2 years', '1 year', '6 months', '3 months', '1 month', '1 week'

# markets = choose_by_length(0, 10000)

print(f"Testing {len(markets)} pairs")
if len(markets) <= 10:
    print(markets)

# lookback window options: '4 years', '3 years', '2 years', '1 year', '6 months', '3 months', '1 month', '1 week'
in_production = True
trader = components.Trader(
    markets,
    dyn_weight_lb='1 week',
    fc_weighting=False,
    port_weights='flat',
    strat_list=strategies,
    keep_records=in_production,
    leverage=2,
    live=in_production
)
trader.run_backtests(
    window='1 year',
    show_stats=True,
    plot_rtns=False,
    plot_forecast=False,
    plot_sharpe=False,
    plot_pnls=False,
    inspect_substrats=False
)
trader.run_trading()

all_end = perf_counter()
print(f"elapsed time: {all_end - all_start:.1f}s")
print("\n****************************************************************************")
