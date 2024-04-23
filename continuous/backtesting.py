from continuous import components

markets = [
    'BTCUSDT',
    'SOLUSDT', 'DUSKUSDT', 'ETHUSDT', 'ROSEUSDT', 'NEARUSDT', 'OMUSDT', 'AVAXUSDT', 'YGGUSDT',
    'ACEUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PENDLEUSDT', 'TUSDT',
]

strategies = [
    'srsirev',
    # 'rsirev',
    # 'chanbreak',
    # 'ichitrend',
    # 'emaroc',
    # 'hmaroc'
    ]
# lookback window options: '4 years', '3 years', '2 years', '1 year', '6 months', '3 months', '1 month', '1 week'
trader = components.Trader(
    markets,
    dyn_weight_lb='1 week',
    fc_weighting=False,
    port_weights='flat',
    strat_list=strategies,
    keep_records=False,
    leverage=10,
    live=False
)
trader.run_backtests(
    window='3 years',
    show_stats=True,
    plot_rtns=False,
    plot_forecast=True,
    plot_sharpe=False,
    plot_pnls=True,
    inspect_substrats=False
)