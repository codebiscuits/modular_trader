# from wootrade import Client as Client_w
from binance.client import Client as Client_b
import binance.exceptions as bx
import binance.enums as be
import mt.resources.keys as keys
from datetime import datetime, timedelta, timezone
import polars as pl
import polars.selectors as cs
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics as stats
from pathlib import Path
from pprint import pprint
import json
from json import JSONDecodeError
import time
from decimal import Decimal
from itertools import product
from continuous import strategies

# client = Client_w(keys.woo_key, keys.woo_secret, keys.woo_app_id, testnet=True)
client = Client_b(keys.bPkey, keys.bSkey)


def calc_perf(s: pl.Series, window: int = 8760) -> dict:
    """calculates 1 year rolling performance statistics on any cummulative pnl series.
    since my pnl series are all in the form of pct changes already, i don't transform them with pct_change in the
    calculations, just diff"""

    ann_mean = s.diff().mean() * 8760
    ann_std = s.diff().std() * 92
    if ann_std:
        ann_sharpe = ann_mean / ann_std
    else:
        ann_sharpe = 0

    dyn_mean = s.diff().ewm_mean(span=window, min_periods=100)[-1] * 8760
    dyn_std = s.diff().ewm_std(span=window, min_periods=100)[-1] * 92
    if dyn_std:
        dyn_sharpe = dyn_mean / dyn_std
    else:
        dyn_sharpe = 0

    dd = (s - s.cum_max()) / s.cum_max()
    max_dd = dd.rolling_min(len(s), min_periods=1)#[-1]
    mean_dd = dd.rolling_mean(len(s), min_periods=1)#[-1]

    # px.line(
    #     data_frame=pl.DataFrame(
    #         data={
    #             'drawdown': dd,
    #             'max_drawdown': max_dd,
    #             'mean_drawdown': mean_dd
    #         }
    #     ),
    #     title='drawdown'
    # ).show()

    return {
        'total_rtn': s[-1],
        'sharpe': ann_sharpe,
        'mean_rtn': ann_mean,
        'std_rtn': ann_std,
        'dyn_sharpe': dyn_sharpe,
        'dyn_mean_rtn': dyn_mean,
        'dyn_std_rtn': dyn_std,
        'skew': s.skew(),
        'max_dd': max_dd[-1],
        'avg_dd': mean_dd[-1]
    }


def print_stats(name: str, d: dict) -> None:
    print(f"{name} Total Rtn: {d['total_rtn']:.1f}x, "
          f"Sharpe: {d['sharpe']:.2f}, "
          f"Mean Rtn: {d['mean_rtn']:.1%}, "
          f"Stdev: {d['std_rtn']:.2f}, "
          f"Skew: {d['skew']:.2f}, "
          f"Max dd: {d['max_dd']:.1%}, "
          f"Avg dd: {d['avg_dd']:.1%}")


def match_lengths(data_in: dict) -> pl.DataFrame:
    max_len = max([len(f) for f in data_in.values()])

    forecasts = pl.DataFrame()
    for k, v in data_in.items():
        fills = 1.0 if 'pnl' in k else 0.0
        forecasts = forecasts.with_columns(
            pl.Series([fills] * (max_len - len(v)), dtype=pl.Float64).extend(v).alias(k))

    return forecasts


def calc_dyn_weight(s: pl.Series, lookback=8760) -> pl.Series:
    """calculates rolling sharpe ratio, annualised for hourly data"""

    ann_mean = s.pct_change().ewm_mean(span=lookback, min_periods=168) * 8760
    ann_std = (s.pct_change().ewm_std(span=lookback, min_periods=168) * 92).clip(lower_bound=0.1)

    # px.histogram(y=ann_std).show()

    return (ann_mean / ann_std).fill_null(0).fill_nan(0).clip(lower_bound=0.5) - 0.5


def step_round(num: float, step: str) -> str:
    """rounds down to any step size"""

    if not float(step):
        return str(num)
    num = Decimal(num)
    step = Decimal(step)

    return str(math.floor(num / step) * step)


class Trader:
    """this class loads and distributes all data that relates to the trading account and the current trading session
    and handles record-keeping. it also instantiates the Coin objects which handle analysis for each trading pair"""

    lookbacks = {'4 years': 35040, '3 years': 26280, '2 years': 17520, '1 year': 8760,
                 '6 months': 4380, '3 months': 2180, '1 month': 720, '1 week': 168}

    def __init__(self, markets: list[str], dyn_weight_lb: str, fc_weighting: bool, port_weights: str,
                 strat_list: list, keep_records: bool = True, leverage: float | str = 1.0, live=False):
        self.now_start = datetime.now().timestamp()
        print(f"Trader initialised at "
              f"{datetime.fromtimestamp(self.now_start, tz=timezone.utc).strftime('%d/%m/%Y %H:%M:%S')} UTC")
        self.markets = markets
        self.dyn_weight_lb = self.lookbacks[dyn_weight_lb]
        self.fc_weighting = fc_weighting
        self.port_weights = port_weights
        self.keep_records = keep_records
        self.target_lev = leverage
        self.live = live
        self.buffer = 0.01  # this is the minimum position change for a trade to be triggered
        self.min_transaction = 10.0  # this is the minimum change in USDT for a trade to be triggered
        self.coins = {m: Coin(m, self.dyn_weight_lb, self.fc_weighting, self.port_weights, strat_list) for m in self.markets}
        self.create_strats()
        self.mkt_data_path = Path('/home/ross/coding/modular_trader/market_data')
        self.mkt_data_path.mkdir(parents=True, exist_ok=True)
        self.records_path = Path('/home/ross/coding/modular_trader/continuous/records')
        self.records_path.mkdir(parents=True, exist_ok=True)

    def run_backtests(self, window, show_stats: bool = False, plot_rtns: bool = False, plot_forecast: bool = False,
                      plot_sharpe: bool = False, plot_pnls: bool = False, inspect_substrats: bool = False):
        for coin in self.coins.values():
            coin.get_raw_forecasts()
            coin.combine_forecasts(inspect=inspect_substrats)
        self.stats = {}
        self.backtest_data = self.get_coin_data()
        self.pnl: pl.DataFrame = self.backtest_portfolio(self.lookbacks[window])

        self.stats = calc_perf(self.pnl['combined_cum_pnl'], self.dyn_weight_lb)
        if show_stats:
            print_stats(self.port_weights, self.stats)
        if plot_rtns:
            px.line(self.backtest_data.select(cs.contains('price')).tail(self.lookbacks[window])).show()
        if plot_forecast:
            fc_col = 'dyn_forecast' if self.port_weights == 'perf' else 'raw_forecast'
            px.line(self.backtest_data.select(cs.contains(fc_col)).tail(self.lookbacks[window])).show()
        if plot_sharpe:
            px.line(self.backtest_data.select(cs.contains('dyn_sharpe')).tail(self.lookbacks[window]),
                    title='dynamic sharpe').show()
        if plot_pnls:
            symbols = self.coins.keys() if len(self.coins) < 10 else len(self.coins)
            title = f"Pairs: {symbols}, dwlb: {self.dyn_weight_lb}, weighting: {self.port_weights}, leverage: {self.target_lev}"
            px.line(y=self.pnl['combined_cum_pnl'], title=title, log_y=True).show()

    def run_trading(self):
        # fetch exchange data
        self.client = Client_b(keys.bPkey, keys.bSkey, testnet=False)
        self.info = self.client.get_exchange_info()
        self.acct = self.client.get_account()
        self.m_acct = self.client.get_margin_account()
        self.spreads = self.binance_spreads()

        # process exchange data
        self.capital: dict = self.account_bal()
        self.asset_bals = self.get_asset_bals()
        self.max_loan()
        self.fees = self.check_fees()
        self.top_up_bnb(20)
        self.spreads_df = self.save_spreads()
        self.get_pairs_info()

        # calculate positions
        self.num_trades = 0
        self.final_sizes = self.get_pos_sizes()
        self.flat_allocations, self.flat_target_pos = self.get_positions('flat', self.keep_records)
        self.lin_allocations, self.lin_target_pos = self.get_positions('lin', self.keep_records)
        self.perf_allocations, self.perf_target_pos = self.get_positions('perf', self.keep_records)
        self.current_positions = self.get_current_pos()  # doesn't include USDT, for all holdings look at asset_bals

        # trade
        self.targets = (
            self.flat_target_pos if self.port_weights == 'flat'
            else self.lin_target_pos if self.port_weights == 'lin'
            else self.perf_target_pos
        )
        trade_diffs = self.pos_diffs(self.targets)
        self.execute_trades(trade_diffs)

        # update info and print summary
        self.m_acct = self.client.get_margin_account()
        self.asset_bals = self.get_asset_bals()
        self.capital: dict = self.account_bal()
        self.actual_lev = (self.capital['usdt_debt'] / self.capital['usdt_net']) + 1
        self.actual_exposure = (
                (self.capital['usdt_gross'] - (self.asset_bals['USDT']['free'] - self.capital['usdt_debt']))
                / self.capital['usdt_gross']
        )
        if self.keep_records:
            self.log_info()
        self.print_info()

    # def take_profit(self):
    #     """this function checks whether the account has made profit over the last week and, if so, takes the necessary steps to secure some profit"""
    #
    #     # only run the function if it's midnight on monday morning
    #     if not ((datetime.now().weekday() == 0) and (datetime.now().hour == 0)):
    #         return
    #
    #     # load the account balance records
    #     try:
    #         with open(self.records_path / 'session.json', 'r') as file:
    #             records = json.load(file)
    #     except (FileNotFoundError, TypeError, JSONDecodeError):
    #         records = []
    #
    #     # extract the record from exactly one week ago and skip the rest if there isn't one
    #     week_ago = (datetime.now() - timedelta(weeks=1)).timestamp()
    #     records = [
    #         {'timestamp': v['timestamp'], 'usdt_net': v['usdt_net'], 'btc_net': v['btc_net']}
    #         for v in records.values()
    #         if v[(week_ago + 3500) >= 'timestamp' >= (week_ago - 3500)]
    #     ]
    #     if not records:
    #         return
    #
    #     # calculate how much the balance has changed in a week in btc and usdt terms
    #     usdt_profit = (self.capital['usdt_net'] - records[0]['usdt_net']) / records[0]['usdt_net']
    #     btc_profit = (self.capital['btc_net'] - records[0]['btc_net']) / records[0]['btc_net']
    #
    #     # skip the rest if the balance went down in either denomination
    #     if (usdt_profit <= 0) or (btc_profit <= 0):
    #         return
    #
    #     # find out how much can be withdrawn at current leverage ratio
    #     max_transfer = float(client.get_max_margin_transfer(asset='USDT')['amount'])
    #     target_withdraw = self.capital['usdt_net'] * usdt_profit * 0.2
    #
    #     if max_transfer < target_withdraw:
    #         # think of a way to programmatically reduce target_lev that can persist from one session to the next

    def create_strats(self):
        for coin in self.coins.values():
            coin.add_strats()

    def save_record(self, data, name):
        now = int(datetime.now().timestamp())
        new_data = data.copy()
        new_data['timestamp'] = now

        try:
            with open(self.records_path / name, 'r') as file:
                old_data = json.load(file)
        except (FileNotFoundError, TypeError, JSONDecodeError):
            old_data = []

        old_data.append(new_data)

        with open(self.records_path / name, 'w') as file:
            json.dump(old_data, file)

    def get_coin_data(self):
        backtests = {}

        for coin in self.coins.values():
            backtests[f"{coin.market}_price"] = coin.pnls['close']
            backtests[f"{coin.market}_returns"] = coin.pnls['returns']
            backtests[f"{coin.market}_raw_forecast"] = coin.pnls['raw_forecast']
            backtests[f"{coin.market}_raw_pnl"] = coin.pnls['raw_pnl']
            backtests[f"{coin.market}_raw_cum_pnl"] = coin.pnls['raw_cum_pnl']
            backtests[f"{coin.market}_dyn_forecast"] = coin.pnls['dyn_forecast']
            backtests[f"{coin.market}_dyn_pnl"] = coin.pnls['dyn_pnl']
            backtests[f"{coin.market}_dyn_cum_pnl"] = coin.pnls['dyn_cum_pnl']
            backtests[f"{coin.market}_dyn_sharpe"] = coin.pnls['dyn_sharpe']

            self.stats[coin.market] = coin.stats

        return match_lengths(backtests)

    def backtest_portfolio(self, window):

        if self.port_weights == 'flat':
            port_pnl: pl.DataFrame = self.backtest_data.select(cs.contains('raw_pnl'))
            port_pnl = ((port_pnl - 1) * self.target_lev) + 1
            port_pnl = port_pnl.with_columns(port_pnl.mean_horizontal().alias('combined_pnl'))

        elif self.port_weights == 'linear':
            port_pnl: pl.DataFrame = self.backtest_data.select(cs.contains('raw_pnl'))
            port_pnl = ((port_pnl - 1) * self.target_lev) + 1
            n = range(1, port_pnl.shape[1] + 1)  # number of coins
            weights = sorted([x / sum(n) for x in n], reverse=True)

            for coin, weight in zip(self.coins, weights):
                port_pnl = port_pnl.with_columns(
                    pl.col(f"{coin}_raw_pnl").mul(weight).alias(f"{coin}_raw_pnl")
                )

            port_pnl = port_pnl.with_columns(port_pnl.sum_horizontal().alias('combined_pnl'))

        elif self.port_weights == 'perf':
            port_pnl: pl.DataFrame = self.backtest_data.select(cs.contains('dyn_pnl'))
            port_pnl = ((port_pnl - 1) * self.target_lev) + 1
            port_pnl = port_pnl.with_columns(port_pnl.mean_horizontal().alias('combined_pnl'))

        elif type(self.port_weights) == list:
            port_pnl: pl.DataFrame = self.backtest_data.select(cs.contains('raw_pnl'))
            port_pnl = ((port_pnl - 1) * self.target_lev) + 1
            for coin, weight in zip(self.coins, self.port_weights):
                port_pnl = port_pnl.with_columns(
                    pl.col(f"{coin}_raw_pnl").mul(weight).alias(f"{coin}_raw_pnl")
                )

            port_pnl = port_pnl.with_columns(port_pnl.sum_horizontal().alias('combined_pnl'))

        else:
            port_pnl = None

        port_pnl = port_pnl.tail(window)
        port_pnl = port_pnl.with_columns(pl.col('combined_pnl').cum_prod().alias('combined_cum_pnl'))

        return port_pnl

    def get_pos_sizes(self):
        # calculate flat weights
        flat_weight = 1 / len(self.markets)

        # calculate linear weights
        n = range(1, len(self.markets) + 1)
        lin_weights = sorted([x / sum(n) for x in n], reverse=True)

        # calculate performance-based weights
        sharpes = self.backtest_data.select(cs.contains('dyn_sharpe')).to_dicts()[-1]
        total_perf_weight = sum([max(s, 0) for s in sharpes.values()]) + 1
        perf_weights = {k: v / total_perf_weight for k, v in sharpes.items()}

        final_sizes = {}
        sizes = self.backtest_data.select(cs.contains('raw_forecast')).to_dicts()[-1]
        for n, m in enumerate(self.markets):
            final_sizes[m] = {
                'size': sizes[f"{m}_raw_forecast"],
                'flat_weight': flat_weight,
                'lin_weight': lin_weights[n],
                'perf_weight': perf_weights[f"{m}_dyn_sharpe"]
            }

        self.save_record(final_sizes, 'final_sizes.json')

        return final_sizes

    def get_positions(self, weighting: str, log: bool):

        # percentage of total capital to allocate to each asset
        allocations = {k: v['size'] * v[f'{weighting}_weight'] for k, v in self.final_sizes.items() if k != 'timestamp'}

        # usdt value to allocate to each asset
        positions = {
            k: v['size'] * v[f'{weighting}_weight'] * self.capital['usdt_net'] * self.target_lev
            for k, v in self.final_sizes.items()
            if k != 'timestamp'
        }

        if log:
            self.save_record(allocations, f"{weighting}_allocations.json")
            self.save_record(positions, f"{weighting}_quote_pos.json")

        return allocations, positions

    def get_current_pos(self):
        current_pos = {f"{k}USDT": v['usdt_value'] for k, v in self.asset_bals.items() if k not in ['USDT', 'BNB']}

        return current_pos

    def pos_diffs(self, target_pos):
        all_pos = list((self.current_positions | target_pos).keys())

        all_trades = {}
        for pos in all_pos:
            old = self.current_positions.get(pos, 0.0)
            new = target_pos.get(pos, 0.0)
            min_diff = max(self.min_transaction, self.buffer * self.capital['usdt_gross'])

            real_difference = new - old
            proportional_difference = ((new - old) / old) if old else 1.0
            # print(f"pos_diffs: {pos} - {old = }, {new = }, {real_difference = }, {proportional_difference = }")
            if abs(real_difference) < min_diff:
                continue

            all_trades[pos] = {
                'long_diff': max(new, 0) - max(old, 0),
                'short_diff': min(new, 0) - min(old, 0)
            }

        return all_trades

    def execute_trades(self, diff):

        print("\ndiff:")
        pprint(diff)
        print('')

        # calculate total usdt borrow need
        total_long_diff = sum([x['long_diff'] for x in diff.values()]) - self.asset_bals['USDT']['free']

        # IMPORTANT NOTE - the diff values are NOTIONAL. When I trigger a buy order, i'm buying with USDT so i use quote
        # sizes for everything, but when i sell, borrow or repay, it's the base size i need to use for all the orders.
        # So even though im passing quote sizes from the diff to the execution functions, i need to calculate base_size
        # as soon as i get into the funcs and use that. OBviously in the case of USDT, quote size is base size so no
        # conversion necessary there

        # borrow USDT as needed
        if total_long_diff > 0:
            total_long_diff *= 1.1
            # print(f"\n\nBorrow {total_long_diff:.2f} USDT")
            loan = self.borrow('USDT', total_long_diff)

        # reduce existing positions as needed
        for asset, change in diff.items():
            if change['long_diff'] < 0:
                # print(f"\n\nSell {abs(change['long_diff']):.2f} USDT of {asset}")
                self.sell(asset, abs(change['long_diff']))
            elif change['short_diff'] > 0:
                # print(f"\n\nBuy {change['short_diff']:.2f} USDT of {asset}")
                self.buy(asset, change['short_diff'])

        # increase existing positions as needed
        for asset, change in diff.items():
            if change['long_diff'] > 0:
                # print(f"\n\nBuy {change['long_diff']:.2f} USDT of {asset}")
                self.buy(asset, change['long_diff'])
            elif change['short_diff'] < 0:
                # print(f"\n\nSell {abs(change['short_diff']):.2f} USDT of {asset}")
                borrowed = self.borrow(asset[:-4], abs(change['short_diff']))
                self.sell(asset, abs(change['short_diff']))

        # calculate how much USDT to keep in case shorts need to be covered and there's no available borrow
        self.total_short_exposure = abs(sum(
            [bal['usdt_value'] for asset, bal in self.asset_bals.items()
             if (asset not in ['BNB', 'USDT']) and (bal['usdt_value'] < 0)]
        ))

        # repay any unnecessary debts
        for asset, values in self.asset_bals.items():
            if values['free'] and (values['borrowed'] or values['interest']):
                owed = values['borrowed'] + values['interest']
                repay_amount = round(min(values['free'], owed), 5)
                # print(f"\n\nRepay {repay_amount} {asset}")
                self.repay(asset, repay_amount)

    def ignore_repay(self, asset, repay_amount):
        """returns True if the repay amount is too small to be worth a transaction."""

        trading = f"{asset}USDT" in self.markets

        if asset == 'USDT':
            repay_threshold = max(25.0, self.total_short_exposure * 0.25)
            repay_value = repay_amount
        elif (asset != 'BNBUSDT') and not trading:
            return False  # if its what remains of an asset i used to trade, just get rid of it
        else:
            asset_price = self.pairs_data[f'{asset}USDT']['price']
            repay_threshold = 10
            repay_value = repay_amount * asset_price
            # print(f"{repay_amount} {asset} at price: {asset_price:.2f} USDT = {repay_value:.2f} USDT value")

        if repay_value < repay_threshold:
            print(f"ignore {repay_amount} {asset} repay, only worth {repay_value:.2f} USDT")

        return repay_value < repay_threshold

    def print_info(self):
        # all_pos = pl.from_dicts([self.lin_allocations, self.perf_allocations])

        pwm = {'flat': 'flat', 'lin': 'linear', 'perf': 'performance-based'}
        print(f"Portfolio weighting method: {pwm[self.port_weights]}")
        print(f"\n{self.target_lev = }")
        print(f"Actual Leverage: {self.actual_lev:.2f} (represents level of debt)")
        print(f"Volatility Exposure: {self.actual_exposure:.1%} (how much of available capital is in trades)")
        print(f"Current Buffer: {self.buffer:.0%}")
        print(f"Maker Fee: {self.fees['maker'] * 10000:.1f}bps, Taker Fee: {self.fees['taker'] * 10000:.1f}bps")
        print("Capital stats:")
        pprint(self.capital)
        print(f"Portfolio: {self.markets}")

        print('Current Backtest Results:')
        print_stats('Performance Stats:', self.stats)
        # print_stats('weighted_perf', self.perf_stats)

        sorted_target_pos = dict(sorted(self.targets.items()))
        sorted_asset_bals = dict(sorted(self.asset_bals.items()))
        print('\nTarget Holdings (USDT):')
        print({k[:-4]: f"{v:.2f}" for k, v in sorted_target_pos.items()})
        print("New Holdings (USDT):")
        print({k: f"{v['usdt_value']:.2f}" for k, v in sorted_asset_bals.items()})
        print("New Holdings as % of net capital:")
        print({k: f"{v['pct']:.1%}" for k, v in sorted_asset_bals.items()})
        print("New Holdings as % of gross capital:")
        print({k: f"{v['adjusted_pct']:.1%}" for k, v in sorted_asset_bals.items()})

    def log_info(self):
        new_data = dict(
            timestamp=self.now_start,
            fc_weighting=self.fc_weighting,
            port_weights=self.port_weights,
            usdt_net=self.capital['usdt_net'],
            btc_net=self.capital['btc_net'],
            max_leverage=self.target_lev,
            real_leverage=self.actual_lev,
            volatility_exposure=self.actual_exposure,
            buffer=self.buffer,
            flat_allocations=self.flat_allocations,
            lin_allocations=self.lin_allocations,
            perf_allocations=self.perf_allocations,
            bt_sharpe=self.stats['sharpe'],
            trades=self.num_trades,
        )

        try:
            with open(self.records_path / 'session.json', 'r') as file:
                old_data = json.load(file)
        except (FileNotFoundError, TypeError, JSONDecodeError):
            old_data = []

        old_data.append(new_data)

        with open(self.records_path / 'session.json', 'w') as file:
            json.dump(old_data, file)

    ###################################################### binance funcs ###################################################

    def get_pairs_info(self):

        not_pairs = ['GBPUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT',
                     'USDCUSDT', 'PAXUSDT', 'COCOSUSDT', 'SUSDUSDT', 'USDPUSDT',
                     'USTUSDT']

        self.pairs_data = {}
        symbols = self.info['symbols']
        for sym in symbols:
            pair = sym.get('symbol')
            right_quote = sym.get('quoteAsset') == 'USDT'
            allowed = pair not in not_pairs
            margin = sym.get('isMarginTradingAllowed')

            if right_quote and margin and allowed:
                base_asset = sym.get('baseAsset')
                oco_allowed = sym['ocoAllowed']
                quote_order_qty_allowed = sym['quoteOrderQtyMarketAllowed']

                if sym.get('status') != 'TRADING':
                    continue

                for f in sym['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size = f['tickSize']
                    elif f['filterType'] == 'LOT_SIZE':
                        step_size = f['stepSize']
                    elif f['filterType'] == 'NOTIONAL':
                        min_size = f['minNotional']
                    elif f['filterType'] == 'MAX_NUM_ALGO_ORDERS':
                        max_algo = f['maxNumAlgoOrders']

                self.pairs_data[pair] = dict(
                    base_asset=base_asset,
                    # cg_symbol=cg_symbol,
                    spread=self.spreads.get(pair),
                    # spread_rank=self.current_spreads.loc[pair, 'rank'],
                    margin_allowed=margin,
                    price_tick_size=tick_size,
                    lot_step_size=step_size,
                    min_notional=min_size,
                    max_algo_orders=max_algo,
                    oco_allowed=oco_allowed,
                    qoq_allowed=quote_order_qty_allowed,
                    spot_orders=[],
                    margin_orders=[],
                )

        prices = {x['symbol']: float(x['price']) for x in self.client.get_all_tickers()}
        for p in self.pairs_data:
            self.pairs_data[p]['price'] = prices[p]

    def binance_spreads(self, quote: str = 'USDT') -> dict[str: float]:
        """returns a dictionary with pairs as keys and current average spread as values"""

        def parse_ob_tickers(tickers):
            s = {}
            for t in tickers:
                if t.get('symbol')[-1 * length:] == quote:
                    pair = t.get('symbol')
                    bid = float(t.get('bidPrice'))
                    ask = float(t.get('askPrice'))
                    if bid and ask:
                        spread = ask - bid
                        mid = (ask + bid) / 2
                        s[pair] = spread / mid

            return s

        length = len(quote)
        avg_spreads = {}

        s_1 = parse_ob_tickers(self.client.get_orderbook_tickers())
        time.sleep(1)

        s_2 = parse_ob_tickers(self.client.get_orderbook_tickers())
        time.sleep(1)

        s_3 = parse_ob_tickers(self.client.get_orderbook_tickers())
        time.sleep(1)

        for k in s_1:
            avg_spreads[k] = stats.median([s_1.get(k), s_2.get(k), s_3.get(k)])

        return avg_spreads

    def save_spreads(self):
        spreads_path = self.mkt_data_path / 'spreads.json'
        spreads_path.touch(exist_ok=True)

        with open(spreads_path, 'r') as file:
            try:
                spreads_data = json.load(file)
            except json.JSONDecodeError:
                spreads_data = {}

        timestamp = int(datetime.now(timezone.utc).timestamp())
        spreads_data[timestamp] = self.spreads

        with open(spreads_path, 'w') as file:
            json.dump(spreads_data, file)

        print(f'\nsaved spreads to {spreads_path}\n')

        # return pl.from_dict(spreads_data)

    def top_up_bnb(self, usdt_size: int) -> dict:
        """checks net BNB balance and interest owed, if net is below the threshold,
        buys BNB then repays any interest"""

        # check balances
        free_bnb = self.asset_bals['BNB']['free']
        interest = self.asset_bals['BNB']['interest']
        free_usdt = self.asset_bals['USDT']['free']
        net_bnb = free_bnb - interest

        # calculate value
        bnb_price = float(self.client.get_avg_price(symbol='BNBUSDT')['price'])
        bnb_value = net_bnb * bnb_price

        def execute_top_up(usdt_size):
            print('Topping up margin BNB')

            order = self.client.create_margin_order(
                symbol='BNBUSDT',
                side=be.SIDE_BUY,
                type=be.ORDER_TYPE_MARKET,
                quoteOrderQty=usdt_size)

            return order

        # top up if needed
        if bnb_value < usdt_size:
            if free_usdt > usdt_size:
                order = execute_top_up(usdt_size)
            else:
                try:
                    borrowed = self.borrow('USDT', usdt_size)  # only works if live is True
                    if borrowed:
                        order = execute_top_up(min(borrowed, usdt_size))
                except bx.BinanceAPIException as e:
                    print(e.code)
                    print(e.message)
                    print('\nWarning - Margin BNB balance low and not enough USDT to top up\n')
                    order = None
        else:
            order = None

        # repay interest
        try:
            if float(interest) > 0.001:
                # uid weight of 3000. not sure how to keep track of this
                self.client.repay_margin_loan(asset='BNB', amount='0.001')
        except bx.BinanceAPIException as e:
            if e.code == -3015:
                print(" Top up BNB caused an exception trying to repay interest")
                return order
            elif e.code == -3041:
                print("Not enough BNB left to repay interest")
                return order
            else:
                raise e

        return order

    def account_bal(self) -> dict:
        """fetches the total value of the margin account holdings from binance
        and returns it, denominated in USDT"""

        btc_price = float(self.client.get_avg_price(symbol='BTCUSDT')['price'])

        return dict(
            btc_gross=float(self.m_acct['totalAssetOfBtc']),
            btc_debt=float(self.m_acct['totalLiabilityOfBtc']),
            btc_net=float(self.m_acct['totalNetAssetOfBtc']),
            usdt_gross=float(self.m_acct['totalCollateralValueInUSDT']),
            usdt_debt=float(self.m_acct['totalLiabilityOfBtc']) * btc_price,
            usdt_net=float(self.m_acct['totalNetAssetOfBtc']) * btc_price,
            collat_margin_lvl=float(self.m_acct['collateralMarginLevel']),
        )

    def check_fees(self):
        margin_bnb = self.asset_bals['BNB']['free']
        maker = Decimal(self.acct['commissionRates']['maker'])
        taker = Decimal(self.acct['commissionRates']['taker'])

        fees = {}
        fees['maker'] = maker * Decimal(0.75) if margin_bnb else maker
        fees['taker'] = taker * Decimal(0.75) if margin_bnb else taker

        return fees

    def get_asset_bals(self) -> dict[str, dict[str, float]]:
        """creates a dictionary of margin asset balances, stored as floats"""

        # the binance client seems to be caching the account data so i'm trying to instantiate
        temp_client = Client_b(keys.bPkey, keys.bSkey)  # a new object to get fresh data
        self.m_acct = temp_client.get_margin_account()

        user_assets = {
            a['asset']: {
                'free': float(a['free']),
                'borrowed': float(a['borrowed']),
                'interest': float(a['interest']),
                'net_asset': float(a['netAsset'])
            }
            for a in self.m_acct['userAssets']
        }

        user_assets = {
            k: v for k, v in user_assets.items()
            if any([v['free'], v['borrowed'], v['interest'], v['net_asset']])
            or f"{k}USDT" in self.markets
        }

        for asset in user_assets:
            if asset == 'USDT':
                usdt_value = user_assets[asset]['net_asset']
            else:
                asset_price = float(temp_client.get_avg_price(symbol=f'{asset}USDT')['price'])
                usdt_value = user_assets[asset]['net_asset'] * asset_price
            pct = usdt_value / self.capital['usdt_net']
            adj_pct = pct / self.target_lev
            user_assets[asset]['usdt_value'] = usdt_value
            user_assets[asset]['pct'] = pct
            user_assets[asset]['adjusted_pct'] = adj_pct

        return user_assets

    def max_loan(self):
        for asset in self.asset_bals:
            try:
                self.asset_bals[asset]['max_loan'] = float(self.client.get_max_margin_loan(asset=asset)['amount'])
            except bx.BinanceAPIException as e:
                if e.code == -3045:
                    print(f"Can't borrow any {asset} at the moment.")
                self.asset_bals[asset]['max_loan'] = 0

    def log_trade(self, trig_price, quote_size, response):

        records_path = Path("/home/ross/coding/modular_trader/continuous/records/trades.json")
        try:
            with open(records_path, 'r') as records_file:
                old_records = json.load(records_file)
        except (FileNotFoundError, JSONDecodeError):
            old_records = []

        fees = sum(float(x['commission']) for x in response['fills'])
        bnb_price = float(self.client.get_avg_price(symbol='BNBUSDT')['price'])
        usdt_fees = fees * bnb_price

        exe_base_size = float(response['executedQty'])
        exe_quote_size = float(response['cummulativeQuoteQty'])
        avg_price = exe_quote_size / exe_base_size

        first_price = float(response['fills'][0]['price'])
        last_price = float(response['fills'][-1]['price'])
        price_impact = abs(first_price - last_price) / first_price

        old_records.append(
            {
                'timestamp': int(response['transactTime']),
                'action': response['side'],
                'pair': response['symbol'],
                'trigger_price': trig_price,
                'desired_quote_size': quote_size,
                'executed_base_size': response['executedQty'],
                'executed_quote_size': response['cummulativeQuoteQty'],
                'avg_price': avg_price,
                'usdt_fees': usdt_fees,
                'price_impact': price_impact,
                'order_id': response['orderId'],
                'logged_by': 1
            }
        )

        with open(records_path, 'w') as records_file:
            json.dump(old_records, records_file)

        print(f"{response['symbol']} {response['side']} logged")

    def log_trade_2(self, trig_price, quote_size, response):

        records_path = Path("/continuous/records_0/trades.json")
        try:
            with open(records_path, 'r') as records_file:
                old_records = json.load(records_file)
        except (FileNotFoundError, JSONDecodeError):
            old_records = []

        exe_base_size = float(response['executedQty'])
        exe_quote_size = float(response['cummulativeQuoteQty'])
        avg_price = exe_quote_size / exe_base_size

        usdt_fees = exe_quote_size * 0.00075  # without fills data i have to calculate fees from total value

        price_impact = abs(trig_price - avg_price) * 2  # without fills data i have to just estimate impact

        old_records.append(
            {
                'timestamp': int(response['transactTime']),
                'action': response['side'],
                'pair': response['symbol'],
                'trigger_price': trig_price,
                'desired_quote_size': quote_size,
                'executed_base_size': response['executedQty'],
                'executed_quote_size': response['cummulativeQuoteQty'],
                'avg_price': avg_price,
                'usdt_fees': usdt_fees,
                'price_impact': price_impact,
                'order_id': response['orderId'],
                'logged_by': 2
            }
        )

        with open(records_path, 'w') as records_file:
            json.dump(old_records, records_file)

        print(f"{response['symbol']} {response['side']} logged")

    def buy(self, pair, quote_size):
        # print(f"Buy {quote_size:.2f} USDT of {pair}")

        # calculate base size
        price = self.pairs_data[pair]['price']
        base_size = quote_size / price
        base_size = step_round(base_size, self.pairs_data[pair]['lot_step_size'])

        if self.live:
            # check usdt balance before buying
            free_bal = self.asset_bals['USDT']['free']
            # if free_bal < quote_size:
            #     print(f"Reducing {pair} buy size from {quote_size:.2f} USDT to {free_bal:.2f} USDT.")
            buy_size = min(free_bal, quote_size)

            # send order to exchange
            if buy_size > self.min_transaction:
                try:
                    buy_order = self.client.create_margin_order(
                        symbol=pair, side=be.SIDE_BUY, type=be.ORDER_TYPE_MARKET, quoteOrderQty=f"{buy_size:.2f}"
                    )
                    spent = float(buy_order['cummulativeQuoteQty'])
                    bought = float(buy_order['executedQty'])
                    self.asset_bals['USDT']['free'] -= spent
                    self.asset_bals['USDT']['net_asset'] -= spent
                    if pair[:-4] not in self.asset_bals:
                        self.asset_bals[pair[:-4]] = {'free': 0.0, 'borrowed': 0.0, 'interest': 0.0, 'net_asset': 0.0,
                                                      'usdt_value': 0.0, 'pct': 0.0, 'adjusted_pct': 0.0}
                    self.asset_bals[pair[:-4]]['free'] += bought
                    self.asset_bals[pair[:-4]]['net_asset'] += bought
                    self.asset_bals[pair[:-4]]['usdt_value'] += spent
                    pct = spent / self.capital['usdt_net']
                    adj_pct = pct / self.target_lev
                    self.asset_bals[pair[:-4]]['pct'] += pct
                    self.asset_bals[pair[:-4]]['adjusted_pct'] += adj_pct

                    self.num_trades += 1
                    if self.keep_records:
                        self.log_trade(price, buy_size, buy_order)
                except bx.BinanceAPIException as e:
                    print(e.code)
                    print(e.message)

        else:
            now = int(datetime.now(timezone.utc).timestamp())
            price = self.pairs_data[pair]['price']

            buy_order = {'clientOrderId': 'not live',
                         'cummulativeQuoteQty': quote_size,
                         'executedQty': str(base_size),
                         'fills': [{'commission': '0',
                                    'commissionAsset': 'BNB',
                                    'price': str(price),
                                    'qty': str(base_size)}],
                         'isIsolated': False,
                         'orderId': 123456,
                         'origQty': str(base_size),
                         'price': '0',
                         'side': 'BUY',
                         'status': 'FILLED',
                         'symbol': pair,
                         'timeInForce': 'GTC',
                         'transactTime': now,
                         'type': 'MARKET'}

            if self.keep_records:
                self.log_trade(price, quote_size, buy_order)

    def sell(self, pair, quote_size):
        # print(f"Sell {quote_size:.5f} USDT of {pair}")

        price = self.pairs_data[pair]['price']
        base_size = quote_size / price

        if self.live:
            # check free balance of asset before selling
            free_bal = self.asset_bals[pair[:-4]]['free']
            # if free_bal < float(base_size):
            #     print(f"Reducing {pair} sell size from {base_size} to {free_bal}.")
            sell_size = min(free_bal, float(base_size))
            if sell_size * price > self.min_transaction:
                sell_size = step_round(sell_size, self.pairs_data[pair]['lot_step_size'])
                try:
                    # send order to exchange
                    sell_order = self.client.create_margin_order(
                        symbol=pair, side=be.SIDE_SELL, type=be.ORDER_TYPE_MARKET, quantity=sell_size
                    )
                    take = float(sell_order['cummulativeQuoteQty'])
                    sold = float(sell_order['executedQty'])
                    self.asset_bals['USDT']['free'] += take
                    self.asset_bals['USDT']['net_asset'] += take
                    if pair[:-4] not in self.asset_bals:
                        self.asset_bals[pair[:-4]] = {'free': 0.0, 'borrowed': 0.0, 'interest': 0.0, 'net_asset': 0.0,
                                                      'usdt_value': 0.0, 'pct': 0.0, 'adjusted_pct': 0.0}
                    self.asset_bals[pair[:-4]]['free'] -= sold
                    self.asset_bals[pair[:-4]]['net_asset'] -= sold

                    self.num_trades += 1
                    if self.keep_records:
                        self.log_trade(price, float(sell_size) * price, sell_order)
                except bx.BinanceAPIException as e:
                    print(e.code)
                    print(e.message)
        else:
            now = int(datetime.now(timezone.utc).timestamp())

            sell_order = {'clientOrderId': '111111',
                          'cummulativeQuoteQty': f"{price * base_size:.2f}",
                          'executedQty': str(base_size),
                          'fills': [{'commission': '0',
                                     'commissionAsset': 'BNB',
                                     'price': str(price),
                                     'qty': str(base_size)}],
                          'isIsolated': False,
                          'orderId': 123456,
                          'origQty': str(base_size),
                          'price': '0',
                          'side': 'SELL',
                          'status': 'FILLED',
                          'symbol': pair,
                          'timeInForce': 'GTC',
                          'transactTime': now,
                          'type': 'MARKET'}

            if self.keep_records:
                self.log_trade(price, price * base_size, sell_order)

    def borrow(self, asset, quote_size):

        # calculate base size
        if asset == 'USDT':
            base_size = quote_size
        else:
            price = self.pairs_data[f"{asset}USDT"]['price']
            base_size = quote_size / price

        if self.live:
            max_loan = self.asset_bals[asset]['max_loan']
            # print(f"max {asset} loan: {max_loan}")
            # if max_loan < base_size:
            #     print(f"Reducing {asset} borrow amount from {base_size} to {max_loan}")
            base_size = f"{min([base_size, max_loan]):.5f}"

            if float(base_size):
                # print(f"Borrow {base_size} {asset}")
                try:
                    tries = 0
                    loan = self.client.create_margin_loan(asset=asset, amount=base_size)
                    time.sleep(1)
                    details = self.client.get_margin_loan_details(asset=asset, txId=loan['tranId'])
                    status = details['rows'][0]['status']
                    # print(f"attempting to borrow {base_size} {asset}, status: {status}")
                    if status != 'CONFIRMED':
                        while status != 'CONFIRMED' and tries < 10:
                            time.sleep(5)
                            details = self.client.get_margin_loan_details(asset=asset, txId=loan['tranId'])
                            status = details['rows'][0]['status']
                            tries += 1
                            # print(f"{asset} loan status: {status} after {tries} tries.")
                    if details['rows'][0]['status'] == 'CONFIRMED':
                        print(f"log {asset} borrow")
                        # pprint(details)
                        if asset not in self.asset_bals:
                            self.asset_bals[asset] = {}
                        self.asset_bals[asset]['free'] += float(details['rows'][0]['principal'])
                        self.asset_bals[asset]['borrowed'] += float(details['rows'][0]['principal'])
                        return float(details['rows'][0]['principal'])
                    else:
                        return 0.0
                except bx.BinanceAPIException as e:
                    if e.code == -3045:  # the system does not have enough asset now
                        print(f"Problem borrowing {base_size} {asset}, not enough to borrow.")
                    else:
                        print(e.code)
                        print(e.message)
                    return 0.0
        else:  # if not live
            return 0.0

    def repay(self, asset, base_size):

        if self.live:
            free_bal = self.asset_bals[asset]['free']
            owed = self.asset_bals[asset]['borrowed'] + self.asset_bals[asset]['interest']
            # if (free_bal < base_size) or (owed < base_size):
            #     print(f"Reducing {asset} repay from {base_size} to {min(free_bal, owed)}")
            base_size = f"{min([free_bal, base_size, owed]) * 0.995:.5f}"

            if self.ignore_repay(asset, float(base_size)):
                return 0.0

            if float(base_size):
                try:
                    tries = 0
                    repay = self.client.repay_margin_loan(asset=asset, amount=base_size)
                    time.sleep(1)
                    details = self.client.get_margin_repay_details(asset=asset, txId=repay['tranId'])
                    status = details['rows'][0]['status']
                    # print(f"attempting to repay {base_size} {asset}, status: {status}")
                    if status != 'CONFIRMED':
                        while status != 'CONFIRMED' and tries < 10:
                            time.sleep(5)
                            details = self.client.get_margin_repay_details(asset=asset, txId=repay['tranId'])
                            status = details['rows'][0]['status']
                            tries += 1
                            # print(f"{asset} repay status: {status} after {tries} tries.")
                    if details['rows'][0]['status'] == 'CONFIRMED':
                        print(f"log {asset} repay")
                        # pprint(details)
                        if asset not in self.asset_bals:
                            self.asset_bals[asset] = {}
                        self.asset_bals[asset]['free'] -= float(base_size)
                        self.asset_bals[asset]['borrowed'] -= float(base_size)
                        return float(details['rows'][0]['principal'])
                    else:
                        return 0.0
                except bx.BinanceAPIException as e:
                    if e.code == -3041:  # balance is not enough
                        print(f"problem repaying {base_size} {asset}, not enough balance")
                    else:
                        print(e.code)
                        print(e.message)
                    return 0.0


class Coin:
    """This class holds all strategies for a particular trading pair, calculates the combined forecast for that pair,
    and returns it to the trader"""
    # TODO i need to test the correlations between forecasts of all these different settings. chances are some of these
    #  won't be necessary, like the doubled settings on the 4h and the standard settings on the 8h are probably highly
    #  correlated

    timeframes = ['1h', '4h', '8h', '1d', '1w']
    lookbacks = [4, 8, 16, 32, 64, 128, 256]

    vb_mas = ['sma', 'ema', 'hma', 'vwma']
    lbs = [25, 50, 100, 200]
    fmas = [6, 9, 12, 15]
    smas = [0.75]
    v_bands = [
        {'tf': '1h', 'ma_type': 'sma', 'lb': 25, 'fast_ma': 3, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 50, 'fast_ma': 3, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 100, 'fast_ma': 3, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 200, 'fast_ma': 3, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 25, 'fast_ma': 3, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 50, 'fast_ma': 3, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 100, 'fast_ma': 3, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 200, 'fast_ma': 3, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 25, 'fast_ma': 6, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 50, 'fast_ma': 6, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 100, 'fast_ma': 6, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 200, 'fast_ma': 6, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 25, 'fast_ma': 6, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 50, 'fast_ma': 6, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 100, 'fast_ma': 6, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'sma', 'lb': 200, 'fast_ma': 6, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 25, 'fast_ma': 3, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 50, 'fast_ma': 3, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 100, 'fast_ma': 3, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 200, 'fast_ma': 3, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 25, 'fast_ma': 3, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 50, 'fast_ma': 3, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 100, 'fast_ma': 3, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 200, 'fast_ma': 3, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 25, 'fast_ma': 6, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 50, 'fast_ma': 6, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 100, 'fast_ma': 6, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 200, 'fast_ma': 6, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 25, 'fast_ma': 6, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 50, 'fast_ma': 6, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 100, 'fast_ma': 6, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'ema', 'lb': 200, 'fast_ma': 6, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 25, 'fast_ma': 3, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 50, 'fast_ma': 3, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 100, 'fast_ma': 3, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 200, 'fast_ma': 3, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 25, 'fast_ma': 3, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 50, 'fast_ma': 3, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 100, 'fast_ma': 3, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 200, 'fast_ma': 3, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 25, 'fast_ma': 6, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 50, 'fast_ma': 6, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 100, 'fast_ma': 6, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 200, 'fast_ma': 6, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 25, 'fast_ma': 6, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 50, 'fast_ma': 6, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 100, 'fast_ma': 6, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'hma', 'lb': 200, 'fast_ma': 6, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 25, 'fast_ma': 3, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 50, 'fast_ma': 3, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 100, 'fast_ma': 3, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 200, 'fast_ma': 3, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 25, 'fast_ma': 3, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 50, 'fast_ma': 3, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 100, 'fast_ma': 3, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 200, 'fast_ma': 3, 'slow_ma': 160},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 25, 'fast_ma': 6, 'slow_ma': 13},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 50, 'fast_ma': 6, 'slow_ma': 25},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 100, 'fast_ma': 6, 'slow_ma': 50},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 200, 'fast_ma': 6, 'slow_ma': 100},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 25, 'fast_ma': 6, 'slow_ma': 20},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 50, 'fast_ma': 6, 'slow_ma': 40},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 100, 'fast_ma': 6, 'slow_ma': 80},
        {'tf': '1h', 'ma_type': 'vwma', 'lb': 200, 'fast_ma': 6, 'slow_ma': 160},
    ]

    prods = product(vb_mas, lbs, fmas, smas)
    v_bands = [
        {'tf': '1h', 'ma_type': m, 'lb': l, 'fast_ma': f, 'slow_ma': s}
        for m, l, f, s in prods
        # if ((l > s) and (s > f))
    ]
    # print(f"v_bands length: {len(v_bands)}")

    srsi_in = 'vwma_200h'
    srsi_reversals = [
        {'tf': '1h', 'lb': 4, 'input': srsi_in},
        {'tf': '1h', 'lb': 8, 'input': srsi_in},
        {'tf': '1h', 'lb': 16, 'input': srsi_in},
        {'tf': '1h', 'lb': 32, 'input': srsi_in},
        {'tf': '1h', 'lb': 64, 'input': srsi_in},
        {'tf': '1h', 'lb': 128, 'input': srsi_in},
        {'tf': '4h', 'lb': 4, 'input': srsi_in},
        {'tf': '4h', 'lb': 8, 'input': srsi_in},
        {'tf': '4h', 'lb': 16, 'input': srsi_in},
        {'tf': '4h', 'lb': 32, 'input': srsi_in},
        {'tf': '4h', 'lb': 64, 'input': srsi_in},
        {'tf': '4h', 'lb': 128, 'input': srsi_in},
        {'tf': '8h', 'lb': 4, 'input': srsi_in},
        {'tf': '8h', 'lb': 8, 'input': srsi_in},
        {'tf': '8h', 'lb': 16, 'input': srsi_in},
        {'tf': '8h', 'lb': 32, 'input': srsi_in},
        {'tf': '8h', 'lb': 64, 'input': srsi_in},
        {'tf': '8h', 'lb': 128, 'input': srsi_in},
    ]

    rsi_reversals = [
        # {'tf': '1h', 'lb': 2, 'input': 'vwma_1h'},
        # {'tf': '1h', 'lb': 4, 'input': 'vwma_1h'},
        # {'tf': '1h', 'lb': 8, 'input': 'vwma_1h'},
        # {'tf': '1h', 'lb': 16, 'input': 'vwma_1h'},
        # {'tf': '4h', 'lb': 2, 'input': 'vwma_25h'},
        # {'tf': '4h', 'lb': 4, 'input': 'vwma_25h'},
        # {'tf': '4h', 'lb': 8, 'input': 'vwma_25h'},
        # {'tf': '4h', 'lb': 16, 'input': 'vwma_25h'},
        {'tf': '8h', 'lb': 2, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 4, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 8, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 16, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 32, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 64, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 128, 'input': 'vwma_50h'},
        {'tf': '8h', 'lb': 256, 'input': 'vwma_50h'},
        # {'tf': '1d', 'lb': 2, 'input': 'vwma_100h'},
        # {'tf': '1d', 'lb': 4, 'input': 'vwma_100h'},
        # {'tf': '1d', 'lb': 8, 'input': 'vwma_100h'},
        # {'tf': '1d', 'lb': 16, 'input': 'vwma_100h'},
        # {'tf': '1w', 'lb': 2, 'input': 'vwma_200h'},
        # {'tf': '1w', 'lb': 4, 'input': 'vwma_200h'},
        # {'tf': '1w', 'lb': 8, 'input': 'vwma_200h'},
        # {'tf': '1w', 'lb': 16, 'input': 'vwma_200h'},
    ]

    chanbreaks = [
        # {'tf': '1h', 'lb': 2},
        # {'tf': '1h', 'lb': 4},
        # {'tf': '1h', 'lb': 8},
        # {'tf': '1h', 'lb': 16},
        # {'tf': '1h', 'lb': 32},
        # {'tf': '1h', 'lb': 64},
        # {'tf': '1h', 'lb': 128},
        # {'tf': '1h', 'lb': 256},
        # {'tf': '4h', 'lb': 2},
        # {'tf': '4h', 'lb': 4},
        # {'tf': '4h', 'lb': 8},
        # {'tf': '4h', 'lb': 16},
        # {'tf': '4h', 'lb': 32},
        # {'tf': '4h', 'lb': 64},
        # {'tf': '4h', 'lb': 128},
        # {'tf': '4h', 'lb': 256},
        # {'tf': '8h', 'lb': 2},
        # {'tf': '8h', 'lb': 4},
        # {'tf': '8h', 'lb': 8},
        # {'tf': '8h', 'lb': 16},
        {'tf': '8h', 'lb': 32},
        # {'tf': '8h', 'lb': 64},
        # {'tf': '8h', 'lb': 128},
        # {'tf': '8h', 'lb': 256},
        # {'tf': '1d', 'lb': 2},
        # {'tf': '1d', 'lb': 4},
        # {'tf': '1d', 'lb': 8},
        # {'tf': '1d', 'lb': 16},
        # {'tf': '1d', 'lb': 32},
        # {'tf': '1d', 'lb': 64},
        # {'tf': '1d', 'lb': 128},
        # {'tf': '1d', 'lb': 256},
        # {'tf': '1w', 'lb': 2},
        # {'tf': '1w', 'lb': 4},
        # {'tf': '1w', 'lb': 8},
        # {'tf': '1w', 'lb': 16},
        # {'tf': '1w', 'lb': 32},
        # {'tf': '1w', 'lb': 64},
        # {'tf': '1w', 'lb': 128},
    ]

    ichitrends = [
        {'tf': '4h', 'f': 10, 's': 30},
        {'tf': '4h', 'f': 20, 's': 60},
        {'tf': '8h', 'f': 10, 's': 30},
        {'tf': '8h', 'f': 20, 's': 60},
        {'tf': '1d', 'f': 10, 's': 30},
        {'tf': '1d', 'f': 20, 's': 60},
        {'tf': '1w', 'f': 10, 's': 30},
        # {'tf': '1w', 'f': 20, 's': 60},
    ]

    moving_avg_rocs = [
        {'tf': '1h', 'lb': 4}, {'tf': '1h', 'lb': 8}, {'tf': '1h', 'lb': 16}, {'tf': '1h', 'lb': 32},
        {'tf': '1h', 'lb': 64}, {'tf': '1h', 'lb': 128}, {'tf': '1h', 'lb': 256},
        {'tf': '4h', 'lb': 4}, {'tf': '4h', 'lb': 8}, {'tf': '4h', 'lb': 16}, {'tf': '4h', 'lb': 32},
        {'tf': '4h', 'lb': 64}, {'tf': '4h', 'lb': 128}, {'tf': '4h', 'lb': 256},
        {'tf': '8h', 'lb': 4}, {'tf': '8h', 'lb': 8}, {'tf': '8h', 'lb': 16}, {'tf': '8h', 'lb': 32},
        {'tf': '8h', 'lb': 64}, {'tf': '8h', 'lb': 128}, {'tf': '8h', 'lb': 256},
        {'tf': '1d', 'lb': 4}, {'tf': '1d', 'lb': 8}, {'tf': '1d', 'lb': 16}, {'tf': '1d', 'lb': 32},
        {'tf': '1d', 'lb': 64}, {'tf': '1d', 'lb': 128}, {'tf': '1d', 'lb': 256},
        {'tf': '1w', 'lb': 4}, {'tf': '1w', 'lb': 8}, {'tf': '1w', 'lb': 16}, {'tf': '1w', 'lb': 32},
        {'tf': '1w', 'lb': 64}, {'tf': '1w', 'lb': 128}, {'tf': '1w', 'lb': 256},
    ]

    def __init__(self, market, dyn_weight_lb: int, fc_weighting, perf_weighting, strat_list: list):
        self.market = market
        self.dyn_weight_lb = dyn_weight_lb
        self.fc_weighting = fc_weighting
        self.perf_weighting = perf_weighting
        self.strat_list = strat_list
        self.data = self.load_data()
        self.pnls = pl.DataFrame()
        self.stats = {}
        self.strats = {}

    def add_strats(self):
        if 'vbands' in self.strat_list:
            self.strats.update({
                f"vbands_{vb['tf']}_{vb['ma_type']}_{vb['lb']}_{vb['fast_ma']}_{vb['slow_ma']}":
                    strategies.VolatilityBands(self.data, vb['tf'], vb['ma_type'], vb['lb'], vb['fast_ma'], vb['slow_ma'])
                for vb in self.v_bands
            })

        if 'srsirev' in self.strat_list:
            self.strats.update({
                f"srsirev_{rr['tf']}_{rr['lb']}_{rr['input']}":
                    strategies.StochRSIReversal(self.data, rr['tf'], rr['lb'], rr['input'])
                for rr in self.srsi_reversals
            })

        rsirev_input_series = 'vwma_200h'
        if 'rsirev' in self.strat_list:
            self.strats.update({
                f"rsirev_{rr['tf']}_{rr['lb']}_{rr['input']}":
                    strategies.RSIReversal(self.data, rr['tf'], rr['lb'], rr['input'])
                for rr in self.rsi_reversals
            })

        chanbreak_input_series = 'vwma_1h'
        if 'chanbreak' in self.strat_list:
            self.strats.update({
                f"chanbreak_{cb['tf']}_{cb['lb']}_{chanbreak_input_series}":
                    strategies.ChanBreak(self.data, cb['tf'], cb['lb'], chanbreak_input_series)
                for cb in self.chanbreaks
            })

        ichitrend_input_series = 'vwma_1h'
        if 'ichitrend' in self.strat_list:
            self.strats.update({
                f"ichitrend_{it['tf']}_{it['f']}_{it['s']}_{ichitrend_input_series}":
                    strategies.IchiTrend(self.data, it['tf'], it['f'], it['s'], ichitrend_input_series)
                for it in self.ichitrends
            })

        if 'emaroc' in self.strat_list:
            self.strats.update({
                f"emaroc_{x['tf']}_{x['lb']}": strategies.EmaRoc(self.data, x['tf'], x['lb']) for x in self.moving_avg_rocs
            })

        if 'hmaroc' in self.strat_list:
            self.strats.update({
                f"hmaroc_{x['tf']}_{x['lb']}": strategies.HmaRoc(self.data, x['tf'], x['lb']) for x in self.moving_avg_rocs
            })

    def __str__(self):
        return f"{self.market} Coin object: {len(self.strats)}"

    def update_ohlc(self, old_end):
        """first calculate whether to use get_klines or get_historical_klines, then download the necessaru ohlc data"""

        new_start = old_end + timedelta(minutes=5)

        span_periods = (datetime.now(timezone.utc) - new_start) / timedelta(minutes=5)
        if span_periods >= 500:
            new_start = str(new_start)
            klines = client.get_historical_klines(symbol=self.market, interval=be.KLINE_INTERVAL_5MINUTE,
                                                  start_str=new_start)
        else:
            new_start = int(new_start.timestamp()) * 1000
            klines = client.get_klines(symbol=self.market, interval=be.KLINE_INTERVAL_5MINUTE, startTime=new_start)

        cols = {'timestamp': pl.Int64,
                'open': pl.String,
                'high': pl.String,
                'low': pl.String,
                'close': pl.String,
                'base_vol': pl.String,
                'close_time': pl.Int64,
                'quote_vol': pl.String,
                'num_trades': pl.Int64,
                'taker_buy_base_vol': pl.String,
                'taker_buy_quote_vol': pl.String,
                'ignore': pl.String}
        data = pl.from_records(klines, schema=cols)

        data = (
            data
            .with_columns(
                pl.col('timestamp').mul(1000000).cast(pl.Datetime(time_zone='UTC', time_unit='ns')),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('base_vol').cast(pl.Float64),
                pl.col('quote_vol').cast(pl.Float64),
                pl.col('num_trades').cast(pl.Float64),
                pl.col('taker_buy_base_vol').cast(pl.Float64),
                pl.col('taker_buy_quote_vol').cast(pl.Float64),
            )
            .drop(['close_time', 'ignore'])
        )

        return data

    def load_data(self):
        datapath = Path(f"/home/ross/coding/modular_trader/bin_ohlc_5m/{self.market}.parquet")
        df = pl.read_parquet(datapath).set_sorted('timestamp')

        df_new = self.update_ohlc(df.item(-1, 'timestamp'))
        df.extend(df_new)
        df = df.sort('timestamp').set_sorted('timestamp')

        df.write_parquet(datapath)

        df = (
            df.with_columns(
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=12, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=12, min_periods=1))
                .alias('vwma_1h'),
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=12 * 25, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=12 * 25, min_periods=1))
                .alias('vwma_25h'),
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=12 * 50, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=12 * 50, min_periods=1))
                .alias('vwma_50h'),
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=12 * 100, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=12 * 100, min_periods=1))
                .alias('vwma_100h'),
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=12 * 200, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=12 * 200, min_periods=1))
                .alias('vwma_200h'),
            )
        )

        df = (df.group_by_dynamic(pl.col('timestamp'), every='1h').agg(
            pl.first('open'),
            pl.max('high'),
            pl.min('low'),
            pl.last('close'),
            pl.sum('base_vol'),
            pl.sum('quote_vol'),
            pl.sum('num_trades'),
            pl.sum('taker_buy_base_vol'),
            pl.sum('taker_buy_quote_vol'),
            pl.last('vwma_1h'),
            pl.last('vwma_25h'),
            pl.last('vwma_50h'),
            pl.last('vwma_100h'),
            pl.last('vwma_200h'),
        ))

        df = df.sort('timestamp')

        df = df.with_columns(
            pl.col('taker_buy_base_vol').mul(2).sub(pl.col('base_vol')).alias('base_vol_delta'),
            pl.col('close').pct_change().fill_null(0).alias('pct_change_1h')
        )

        # TODO maybe try optimising the rolling_std window, 500 was just a guess
        annualiser = {'1h': 94, '4h': 47, '8h': 33, '1d': 19, '2d': 14, '3d': 11, '1w': 7}['1h']
        df = df.with_columns((pl.col('pct_change_1h').rolling_std(500, min_periods=3) * annualiser).alias(f'dyn_std'))

        return (df.select(['timestamp', 'open', 'high', 'low', 'close', 'quote_vol', 'base_vol_delta', 'pct_change_1h',
                           'dyn_std', 'vwma_1h', 'vwma_25h', 'vwma_50h', 'vwma_100h', 'vwma_200h'])
                .set_sorted('timestamp'))

    def get_raw_forecasts(self):
        """get raw forecasts from each child, put them in a new pl.DataFrame,
        and match the length with the original hourly data"""

        # collect forecasts from sub-strats
        fc_dict = {k: v.forecast for k, v in self.strats.items()}
        max_len = max([len(f) for f in fc_dict.values()])

        # pprint(fc_dict)

        self.raw_forecasts = pl.DataFrame()
        for k, v in fc_dict.items():
            self.raw_forecasts = self.raw_forecasts.with_columns(
                pl.Series([0.0] * (max_len - len(v)), dtype=pl.Float64)
                .extend(v)
                .alias(k))
        # px.line(self.forecasts.tail(35000), title='individual un-weighted forecast').show()

        # match length of forecasts to hourly ohlc data
        if len(self.raw_forecasts) > len(self.data):
            self.raw_forecasts = self.raw_forecasts.tail(len(self.data))
        elif len(self.raw_forecasts) < len(self.data):
            self.data = self.data.tail(len(self.raw_forecasts))

    def forecast_to_returns(self, df, stage):
        """quantises the forecast, calculates proportional turnover and trading costs from trading the quantised
        forecast, then uses that data to calculate a returns series and a cummulative returns series"""
        q = 0.01
        costs = 0.005

        # quantise the forecast
        df = df.with_columns(pl.col(f"{stage}_forecast")
                             .truediv(q).round().mul(q)
                             .shift(n=1, fill_value=0)
                             .alias(f"{stage}_forecast"))

        # calculate turnover and trading costs
        df = df.with_columns(pl.col(f"{stage}_forecast").diff().fill_null(0).alias(f'{stage}_trade_size'))
        df = df.with_columns(pl.col(f'{stage}_trade_size').mul(costs).alias(f'{stage}_trade_costs'))

        # calculate raw pnl from the forecast, the pair's returns and the trading costs
        df = df.with_columns(
            pl.col(f"{stage}_forecast")
            .mul(pl.col('returns'))
            .add(1)
            .mul(pl.lit(1) - pl.col(f'{stage}_trade_costs'))
            .alias(f'{stage}_pnl')
        )

        # cummulate raw pnl series for backtesting
        df = df.with_columns(pl.col(f'{stage}_pnl').cum_prod().fill_null(1.0).alias(f'{stage}_cum_pnl'))

        return df

    def perf_weight_forecast(self, fc: pl.Series):
        """transforms a raw forecast into a performance-weighted dynamic forecast"""

        # put the forecast and the pair's returns series together in a dataframe
        returns = self.data.get_column('pct_change_1h')
        close = self.data.get_column('close')

        temp = match_lengths({'raw_forecast': fc, 'returns': returns, 'close': close})

        temp = self.forecast_to_returns(temp, 'raw')

        # backtest raw pnl series to calculate dynamic sharpe
        temp = temp.with_columns(
            calc_dyn_weight(temp.get_column('raw_cum_pnl'), self.dyn_weight_lb)
            .ewm_mean(span=1000)
            .alias('dyn_sharpe'),
            (calc_dyn_weight(temp.get_column('raw_cum_pnl'), self.dyn_weight_lb) / 2)
            .ewm_mean(span=1000)
            .clip(lower_bound=0, upper_bound=2)
            .alias('dyn_sharpe_lim'),
        )

        # create dynamic forecast from raw forecast and dynamic sharpe
        temp = temp.with_columns(
            pl.col('raw_forecast')
            .mul(
                pl.col('dyn_sharpe_lim')
                .shift(n=1, fill_value=0)
                # .sub(0.5)
            )
            .alias('dyn_forecast')
        )

        return temp

    def inspect_perf_weighting(self, df, name):
        """takes the dataframe from perf_weight_forecast and plots the various series so i can see how that forecast is
        performing dynamically"""

        df = self.forecast_to_returns(df, 'dyn')

        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        fig.add_trace(go.Scatter(y=df['close'], name='price'), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['raw_forecast'], name='raw_forecast'), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['raw_cum_pnl'], name='raw_cum_pnl'), row=3, col=1)
        fig.add_trace(go.Scatter(y=df['dyn_sharpe_lim'], name='dyn_sharpe_lim'), row=4, col=1)
        fig.add_trace(go.Scatter(y=df['dyn_sharpe'], name='dyn_sharpe'), row=4, col=1)
        fig.add_trace(go.Scatter(y=df['dyn_forecast'], name='dyn_forecast'), row=5, col=1)
        fig.add_trace(go.Scatter(y=df['dyn_cum_pnl'], name='dyn_cum_pnl'), row=6, col=1)

        raw_sharpe = calc_perf(df['raw_cum_pnl'])['sharpe']
        dyn_sharpe = calc_perf(df['dyn_cum_pnl'])['sharpe']

        title = f"{name} raw sharpe: {raw_sharpe:.2f}, dynamic sharpe: {dyn_sharpe:.2f}\n"

        fig.update_layout(height=600, width=1200, title_text=title)
        fig.show()

    def get_weighted_forecasts(self, inspect: bool = False):
        """gets the individual raw forecasts from each active sub-strat and turns them into individual dynamic
        forecasts, then puts them all together in a dataframe and returns the dataframe"""

        weighted_forecasts_dict = {}
        for strat_name, strat in self.strats.items():
            fc_df = self.perf_weight_forecast(strat.forecast)
            weighted_forecasts_dict[strat_name] = fc_df["dyn_forecast"]
            if inspect:
                self.inspect_perf_weighting(fc_df, strat_name)

        return match_lengths(weighted_forecasts_dict)

    def process_forecast(self,
                         forecast: pl.Series,
                         long_only: bool = False,
                         standardise: bool = False,
                         normalise: bool = True,
                         flip: bool = False,
                         smooth: bool = False,
                         quantise: float = 0.5,) -> pl.Series:

        # fit range to long and short
        if not long_only:
            forecast = (forecast * 2) - 1
            # self.data = self.data.with_columns(pl.col(self.forecast_col).mul(2).sub(1).alias(self.forecast_col))

        # standardise forecast at local timescale
        if standardise:
            forecast /= self.data['dyn_std']
            # self.data = (
            #     self.data.with_columns(
            #         pl.col(self.forecast_col)
            #         .truediv(pl.col('dyn_std'))
            #         .alias(self.forecast_col)
            #     )
            # )

        if normalise:
            forecast = forecast / forecast.abs().mean()
            # self.data = (
            #     self.data.with_columns(
            #         pl.col(self.forecast_col)
            #         .truediv(pl.col(self.forecast_col).abs().mean())
            #         .alias(self.forecast_col)
            #     )
            # )

        if flip:
            forecast *= -1
            # self.data = (
            #     self.data.with_columns(
            #         pl.col(self.forecast_col)
            #         .mul(-1)
            #         .alias(self.forecast_col)
            #     )
            # )

        if smooth:
            forecast = forecast.ewm_mean(span=3)
            # self.data = (
            #     self.data.with_columns(
            #         pl.col(self.forecast_col)
            #         .ewm_mean(span=3)
            #         .alias(self.forecast_col)
            #     )
            # )

        if quantise:
            q = 0.2
            forecast = ((forecast / q).round()) * q
            # self.data = (
            #     self.data.with_columns(
            #         pl.col(self.forecast_col)
            #         .truediv(quantise)
            #         .round()
            #         .mul(quantise)
            #         .alias(self.forecast_col)
            #     )
            # )

        # clip
        forecast = forecast.clip(lower_bound=-1, upper_bound=1)
        # self.data = (
        #     self.data.with_columns(
        #         pl.col(self.forecast_col)
        #         .clip(lower_bound=-1, upper_bound=1)
        #         .alias(self.forecast_col)
        #     )
        # )

        # return self.data.get_column(self.forecast_col).fill_null(0).ewm_mean(6)
        return forecast

    def combine_forecasts(self, inspect: bool = False):

        if self.fc_weighting:
            self.weighted_forecasts = self.get_weighted_forecasts(inspect)
            forecasts = self.weighted_forecasts
        else:
            forecasts = self.raw_forecasts

        combined_forecast = forecasts.mean_horizontal()

        # i might want to process the combined forecast here
        # combined_forecast = self.process_forecast(combined_forecast)

        # then i want to backtest the combined forecast

        # then, if i'm perf-weighting the portfolio, i would apply perf-weighting to the combined forecast here
        combined = self.perf_weight_forecast(combined_forecast)
        if self.perf_weighting:
            self.pnls = self.forecast_to_returns(combined, 'dyn')
        else:
            self.pnls = self.forecast_to_returns(combined, 'raw')
        # print(f"{self.market} pnls df: {self.pnls.columns}")
        dyn_combined_forecast = self.pnls['dyn_forecast']

        # then i can backtest the perf-weighted portfolio of forecasts in the trader maybe

    # def get_weights(self) -> pl.DataFrame:
    #
    #     # this block backtests each individual sub-forecast so they can be dynamically weighted in the main forecast
    #     scores = {}
    #     for fc in self.strats:
    #         scores[fc] = self.backtest_forecast(fc)[2]
    #
    #     dynamic_weights = pl.DataFrame(scores)
    #     # px.line(dynamic_weights.tail(35000), title='individual forecast weight').show()
    #
    #     # if there are backtests with good sharpe ratios, this will divide everything by the total, so they keep their
    #     # relative weighting but end up combining to 1. however, if the backtests are all crap and average to less than
    #     # 1, dividing by the total would actually increase allocation to all of them. lower_bound=thresh ensures that i
    #     # leave them with small weightings in these cases. This way, when times are bad I'm not fully exposed.
    #     thresh = len(dynamic_weights.columns)
    #     dynamic_weights = dynamic_weights.with_columns(
    #         dynamic_weights.sum_horizontal()
    #         .clip(lower_bound=1)
    #         .alias('total_weight')
    #     )
    #     # divide each dynamic weight series by total dynamic weight series and clip
    #     final_weights = pl.DataFrame(
    #         [(col / dynamic_weights.get_column('total_weight')).fill_null(0).clip(lower_bound=0)
    #          for col in dynamic_weights.iter_columns()]
    #     )
    #     # final_weights = dynamic_weights
    #
    #     return final_weights
    #
    # def old_combine_forecasts(self):
    #
    #     weights = self.get_weights()
    #
    #     weighted_forecasts = {}
    #     for fc in self.strats:
    #         weighted_forecasts[fc] = (
    #                 self.forecasts.get_column(fc) * weights.get_column(fc).shift(n=1, fill_value=0)
    #         )
    #     self.weighted_forecasts = pl.DataFrame(weighted_forecasts)
    #
    #     # combine individual sub-strat forecasts into weighted and unweighted coin forecasts
    #     self.forecasts = self.forecasts.with_columns(
    #         self.forecasts.mean_horizontal().alias('flat_size'),
    #         self.weighted_forecasts.sum_horizontal().alias('weighted_size')
    #     )
    #
    #     # # scale forecasts
    #     # self.forecasts = self.forecasts.with_columns(
    #     #     (
    #     #         pl.col('flat_size')
    #     #         .truediv(pl.col('flat_size').abs().mean())
    #     #         .truediv(2)
    #     #         .clip(lower_bound=-1, upper_bound=1)
    #     #         .alias('flat_size')
    #     #     ),
    #     #     (
    #     #         pl.col('weighted_size')
    #     #         .truediv(pl.col('weighted_size').abs().mean())
    #     #         .truediv(2)
    #     #         .clip(lower_bound=-1, upper_bound=1)
    #     #         .alias('weighted_size')
    #     #     ),
    #     # )
    #
    #     # backtest weighted and unweighted combined forecasts
    #     flat_scores = self.backtest_forecast('flat_size')
    #     weighted_scores = self.backtest_forecast('weighted_size')
    #
    #     # store forecast, returns, cum returns and dynamic sharpe for weighted and unweighted forecasts
    #     self.pnls = self.pnls.with_columns(
    #         self.forecasts['flat_size'].alias('flat_size'),
    #         flat_scores[0].alias('flat_raw_pnl'),
    #         flat_scores[1].alias('flat_cum_pnl'),
    #         flat_scores[2].alias('flat_dyn_sharpe'),
    #         self.forecasts['weighted_size'].alias('weighted_size'),
    #         weighted_scores[0].alias('weighted_raw_pnl'),
    #         weighted_scores[1].alias('weighted_cum_pnl'),
    #         weighted_scores[2].alias('weighted_dyn_sharpe'),
    #         weighted_scores[3].alias('pair_rtns'),
    #     )
    #
    #     self.stats['flat'] = flat_scores[4]
    #     self.stats['weighted'] = weighted_scores[4]
    #
    #     # px.line(data_frame=self.weighted_forecasts, title="weighted_forecasts").show()
    #     # px.line(data_frame=self.forecasts.select(['flat_size', 'final_size']), title="final forecasts").show()
    #     # px.line(data_frame=self.pnls, title="pnls and sharpes").show()
    #
    # def old_backtest_forecast(self, name: str) -> tuple[pl.DataFrame, dict]:
    #     """backtest the named forecast"""
    #
    #     # put the forecast and the pair's returns series together in a dataframe
    #     fc = self.raw_forecasts.get_column(name)
    #     returns = self.data.get_column('pct_change_1h')
    #     close = self.data.get_column('close')
    #     temp = pl.DataFrame({'forecast': fc, 'returns': returns, 'close': close})
    #
    #     ###########################
    #
    #     q = 0.2
    #     costs = 0.005
    #
    #     # quantise the forecast
    #     temp = temp.with_columns(pl.col('forecast')
    #                              .truediv(q).round().mul(q)
    #                              .shift(n=1, fill_value=0)
    #                              .alias('forecast'))
    #
    #     # calculate turnover and trading costs
    #     temp = temp.with_columns(pl.col('forecast').diff().fill_null(0).alias('trade_size'))
    #     temp = temp.with_columns(pl.col('trade_size').mul(costs).alias('trade_costs'))
    #
    #     # calculate raw pnl from the forecast, the pair's returns and the trading costs
    #     temp = temp.with_columns(
    #         pl.col('forecast')
    #         .mul(pl.col('returns'))
    #         .add(1)
    #         .mul(pl.lit(1) - pl.col('trade_costs'))
    #         .alias(f'{name}_raw_pnl')
    #     )
    #
    #     # cummulate raw pnl series for backtesting
    #     temp = temp.with_columns(pl.col(f'{name}_raw_pnl').cum_prod().fill_null(1.0).alias(f'{name}_cum_pnl'))
    #
    #     ################################
    #
    #     # backtest raw pnl series to calculate dynamic sharpe
    #     temp = temp.with_columns(
    #         calc_dyn_weight(temp.get_column(f'{name}_cum_pnl'), self.dyn_weight_lb)
    #         .alias(f'{name}_dyn_sharpe')
    #     )
    #
    #     # create dynamic forecast from raw forecast and dynamic sharpe
    #     temp = temp.with_columns(
    #         pl.col('forecast')
    #         .mul(pl.col(f'{name}_dyn_sharpe').sub(0.5).clip(lower_bound=0, upper_bound=1))
    #         .alias('dyn_forecast')
    #     )
    #
    #     ################################
    #
    #     # quantise the dynamic forecast
    #     temp = temp.with_columns(pl.col('dyn_forecast')
    #                              .truediv(q).round().mul(q)
    #                              .shift(n=1, fill_value=0)
    #                              .alias('dyn_forecast'))
    #
    #     # re-calculate turnover and trading costs
    #     temp = temp.with_columns(pl.col('dyn_forecast').diff().fill_null(0).alias('trade_size'))
    #     temp = temp.with_columns(pl.col('trade_size').mul(costs).alias('trade_costs'))
    #
    #     # calculate dynamic pnl from the dynamic forecast, the pair's returns, and updated trading costs
    #     temp = temp.with_columns(
    #         pl.col('dyn_forecast')
    #         .mul(pl.col('returns'))
    #         .add(1)
    #         .mul(pl.lit(1) - pl.col('trade_costs'))
    #         .alias(f'{name}_dyn_pnl')
    #     )
    #
    #     # cummulate dynamic pnl series for backtesting
    #     temp = temp.with_columns(pl.col(f'{name}_dyn_pnl').cum_prod().alias(f'{name}_dyn_cum_pnl'))
    #
    #     ###############################
    #
    #     perf_metrics = calc_perf(temp.get_column(f'{name}_dyn_cum_pnl'), self.dyn_weight_lb)
    #     perf_metrics['turnover'] = temp['trade_size'].sum()
    #     perf_metrics['total_costs'] = temp['trade_costs'].sum()
    #
    #     return temp, perf_metrics


