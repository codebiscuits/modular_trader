from continuous import components
import polars as pl
import plotly.express as px
from itertools import product
import json


def run_backtest(p, l, f, w):
    print(f"\nbacktesting: fc: {f}, port weight: {w}, dyn_lb_{p}_lev_{l}")
    # lookback window options: '4 years', '3 years', '2 years', '1 year', '6 months', '3 months', '1 month', '1 week'
    trader = components.Trader(
        markets,
        dyn_weight_lb=p,
        fc_weighting=f,
        port_weights=w,
        strat_list=strategies,
        keep_records=False,
        leverage=l,
        live=False
    )
    trader.run_backtests(
        window=backtest_window,
        show_stats=True,
        plot_rtns=False,
        plot_forecast=False,
        plot_sharpe=False,
        plot_pnls=False,
        inspect_substrats=False
    )

    return trader


def optimise():
    pnl_charts = pl.DataFrame()
    pnl_results = []
    periods = ['1 week', '1 month', '3 months', '6 months', '1 year', '2 years'][0:1]
    levs = [0.5, 1, 2, 3, 4, 5]
    fc = [True, False][1:]
    ports = ['flat', 'perf'][0:1]
    for p, l, f, w in product(periods, levs, fc, ports):
        trader = run_backtest(p, l, f, w)

        pnl_charts = pnl_charts.with_columns(trader.pnl['combined_cum_pnl'].alias(f"{f}_{w}_{p}_{l}"))

        results = dict(
            lb=p,
            fc=f,
            port=w,
            lev=l,
            sharpe=trader.stats['sharpe'],
            total_rtn=trader.stats['total_rtn'],
            max_dd=trader.stats['max_dd'],
            avg_dd=trader.stats['avg_dd'],
        )

        pnl_results.append(results)

    t = f"{strategies}, {len(markets)} pairs."
    px.line(data_frame=pnl_charts, title=t, log_y=True).show()

    with open("/home/ross/coding/modular_trader/continuous/backtest_results.json", "w") as f:
        json.dump(pnl_results, f)


def single_backtest(p, l, f, w):
    trader = run_backtest(p, l, f, w)

    print(
        f"lb: {p}, fc: {f}, port: {w}, lev: {l}, sharpe: {trader.stats['sharpe']:.1f}, "
        f"total_rtn: {trader.stats['total_rtn']:.1f}x, max_dd: {trader.stats['max_dd']:.1%}, "
        f"avg_dd: {trader.stats['avg_dd']:.1%}"
    )
    px.line(y=trader.pnl['combined_cum_pnl'], title=f"{f}_{w}_{p}_{l}", log_y=True).show()


markets = [
    'BTCUSDT',
    'SOLUSDT', 'DUSKUSDT', 'ETHUSDT', 'ROSEUSDT', 'NEARUSDT', 'OMUSDT', 'AVAXUSDT', 'YGGUSDT',
    'ACEUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PENDLEUSDT', 'TUSDT',
]

strategies = [
    'vbands',
    'srsirev',
    # 'rsirev',
    # 'chanbreak',
    'ichitrend',
    # 'emaroc',
    # 'hmaroc'
    ]

dyn_weight_lb = '1 week'
backtest_window = '3 years'

# single_backtest('1 week', 1, False, 'flat')
optimise()
