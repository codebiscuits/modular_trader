from pathlib import Path
import polars as pl
import ind_pl as ind
from binance import Client
from mt.resources import keys
from datetime import datetime
import json
import time

all_start = time.perf_counter()
client = Client(keys.mPkey, keys.mSkey)
print(f"Running screener on {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")

def choose_by_length(minimum: int | str = 27500, maximum: int | str = 420000):
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
        try:
            df = pl.read_parquet(pair_path)
        except pl.exceptions.ComputeError:
            continue
        info[pair_path.stem] = len(df)

    return [p for p, v in info.items() if minimum < v <= maximum]


def resample(df, timeframe):
    df = df.sort('timestamp').set_sorted('timestamp')

    df = (df.group_by_dynamic(pl.col('timestamp'), every=timeframe).agg(
        pl.first('open'),
        pl.max('high'),
        pl.min('low'),
        pl.last('close'),
        pl.sum('base_vol'),
        pl.sum('quote_vol'),
        pl.sum('num_trades'),
        pl.sum('taker_buy_base_vol'),
        pl.sum('taker_buy_quote_vol'),
    ))

    df = df.sort('timestamp')

    return df


def top_heavy(df: pl.DataFrame) -> pl.DataFrame:
    """calculates my 'top heavy' metric, which is a ratio representing the balance of total historic
    volume below the current price relative to total historic volume above the current price"""

    current_price = df.item(-1, 'close')

    above = df.filter(pl.col('close') > current_price)['base_vol'].sum()
    below = df.filter(pl.col('close') < current_price)['base_vol'].sum()

    return above, below, above / below

x_info = client.get_exchange_info()
tick_sizes = {x['symbol']: float(x['filters'][0]['tickSize']) for x in x_info['symbols']}

info = {
    'pair': [],
    'length': [],
    'daily_volume': [],
    'weekly_volume': [],
    'daily_volume_change': [],
    'daily_atr': [],
    'tick_size': [],
    # 'weekly_rsi': [],
    # 'monthly_rsi': [],
    'volume_above_pw': [],  # historic volume above the current price, in multiples of current weekly volume
    'volume_below_pw': [],  # historic volume below the current price, in multiples of current weekly volume
    'top_heavy': [],
    # 'sharpe': [],  # this will have to be backtested
    # 'mcap': [],  # this will have to come from coingecko
    'divisibility': []
}

pairs = choose_by_length()

for pair in pairs:
    pair_path = Path(f"/home/ross/coding/pi_3/modular_trader/bin_ohlc_5m/{pair}.parquet")
    try:
        df = pl.read_parquet(pair_path)
    except FileNotFoundError:
        continue
    weekly_df = resample(df, '1w')
    monthly_df = resample(df, '1m')
    one_day_volume = df['quote_vol'].tail(288).sum()
    seven_day_volume = df['quote_vol'].tail(2016).sum()
    daily_volume_change = df['quote_vol'].ewm_mean(span=288).pct_change(288)[-1]
    daily_atr_pct = ind.atr(df, 48, 288, 'pct').item(-1, 'atr_48_288_pct')
    daily_atr_abs = ind.atr(df, 48, 288, 'abs').item(-1, 'atr_48_288_abs')
    info['pair'].append(pair)
    info['length'].append(len(df))
    info['daily_volume'].append(one_day_volume)
    info['weekly_volume'].append(seven_day_volume)
    info['daily_volume_change'].append(daily_volume_change)
    info['daily_atr'].append(daily_atr_pct)
    info['tick_size'].append(tick_sizes[pair])
    # info['weekly_rsi'].append(weekly_df['close'])
    # info['monthly_rsi'].append(monthly_df['close'])
    info['volume_above_pw'].append(top_heavy(df)[0] / seven_day_volume)
    info['volume_below_pw'].append(top_heavy(df)[1] / seven_day_volume)
    info['top_heavy'].append(top_heavy(df)[2])
    info['divisibility'].append(daily_atr_abs / tick_sizes[pair])

info_df = pl.from_dict(info)
lively_pairs = info_df.filter(
    pl.col('divisibility').gt(80.0),
    pl.col('daily_volume').gt(2_500_000),
    pl.col('length').gt(27_500),
    pl.col('daily_atr').rank(descending=True).lt(50),
)

# load ohlc, resample to 1h, then calculate correlation matrix for all pairs in lively pairs, and record avg correlation as a new stat for each pair
all_closes = {}
for pair in lively_pairs['pair']:
    pair_path = Path(f"/home/ross/coding/pi_3/modular_trader/bin_ohlc_5m/{pair}.parquet")
    try:
        df = pl.read_parquet(pair_path)
    except FileNotFoundError:
        continue
    df = resample(df, '1h')
    all_closes[pair] = df['close']

min_length = min([len(x) for x in all_closes.values()])
all_closes = {k: v.tail(min_length) for k, v in all_closes.items()}
closes_df = pl.DataFrame(all_closes)

new_pairs = (
    lively_pairs
    .with_columns(
        pl.Series(closes_df.corr()
                  .mean()
                  .to_dicts()[0]
                  .values()
                  )
        .alias('avg_correlation')
    )
    .filter(
        pl.col('avg_correlation')
        .rank()
        .lt(16)
    )
    ['pair']
    .to_list()
)

print(new_pairs)

year = datetime.now().year
timestamp = int(datetime.now().timestamp())
port_file = Path(f"records/pairs_{year}.json")

if port_file.exists():
    with open(port_file, 'r') as f:
        portfolio_history = json.load(f)
else:
    portfolio_history = []

with open(port_file, 'w') as f:
    portfolio_history.append({timestamp: new_pairs})
    json.dump(portfolio_history, f)

elapsed = time.perf_counter() - all_start
print(f"Time taken: {elapsed // 60:.0f}m {elapsed % 60:.0f}s\n")