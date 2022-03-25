from pathlib import Path
from pushbullet import Pushbullet
from datetime import datetime

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
now = datetime.now().strftime('%d/%m/%y %H:%M')

not_pairs = ['BNBUSDT', 'GBPUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 
             'USDCUSDT', 'PAXUSDT', 'COCOSUSDT', 'SUSDUSDT', 'USDPUSDT', 
             'USTUSDT']

# constants
params = {'quote_asset': 'USDT', 
          'fr_range': (0.00025, 0.00175), 
          'max_spread': 0.5, 
          'indiv_r_limit': 1.2, 
          'total_r_limit': 20, 
          'target_risk': 0.1, 
          'max_pos': 20}

# ohlc data paths
ohlc_data = None
possible_paths = [Path('/media/coding/ohlc_binance_1h'), 
                  Path('/mnt/pi_2/ohlc_binance_1h')]

for ohlc_path in possible_paths:
    if ohlc_path.exists():
        ohlc_data = ohlc_path
        break
if not ohlc_data:
    note = 'none of the paths for ohlc_data are available'
    print(note)
    pb.push_note(now, note)


# market data paths
market_data = None
poss_paths = [Path('/media/coding/market_data'), 
              Path('/mnt/pi_2/market_data')]

for md_path in poss_paths:
    if md_path.exists():
        market_data = md_path
        break
if not market_data:
    note = 'none of the paths for market_data are available'
    print(note)
    pb.push_note(now, note)


# backtesting results paths
results_data = None
poss_paths = [Path('/media/coding/results'), 
              Path('/mnt/pi_2/results')]

for res_path in poss_paths:
    if res_path.exists():
        results_data = res_path
        break
if not results_data:
    note = 'none of the paths for results_data are available'
    print(note)
    pb.push_note(now, note)
