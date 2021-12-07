from pathlib import Path
from pushbullet import Pushbullet
from datetime import datetime

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
now = datetime.now().strftime('%d/%m/%y %H:%M')

not_pairs = ['GBPUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 
             'USDCUSDT', 'PAXUSDT', 'COCOSUSDT', 'SUSDUSDT', 'USDPUSDT', 
             'ADADOWNUSDT', 'LINKDOWNUSDT', 'BNBDOWNUSDT', 'ETHDOWNUSDT']

# TODO turn this pathfinding into a function that can be run at the start of a
# script, instead of an attribute to import from here

# ohlc data paths
ohlc_data = None
pi2_ohlc_bin = Path('/media/coding/ohlc_binance_1h')
pi_ohlc_bin = Path('/mnt/2tb_ssd/coding/ohlc_binance_1h')
lap_ohlc_bin = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/ohlc_binance_1h')
desk_ohlc_bin = Path('/mnt/pishare/ohlc_binance_1h')
# desk_ohlc_bin = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/ohlc_binance_1h')
for ohlc_path in [pi2_ohlc_bin, pi_ohlc_bin, lap_ohlc_bin, desk_ohlc_bin]:
    if ohlc_path.exists():
        ohlc_data = ohlc_path
        break
if not ohlc_data:
    note = 'none of the paths for ohlc_data are available'
    print(note)
    pb.push_note(now, note)

# market data paths
market_data = None
pi2_md = Path('/media/coding/market_data')
pi_md = Path('/mnt/2tb_ssd/coding/market_data')
lap_md = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/market_data')
# desk_md = Path('/mnt/pishare/market_data')
desk_md = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/market_data')
for md_path in [pi_md, lap_md, desk_md]:
    if md_path.exists():
        market_data = md_path
        break
if not market_data:
    note = 'none of the paths for market_data are available'
    print(note)
    pb.push_note(now, note)

# backtesting results paths
results_data = None
pi2_res = Path('/media/coding/results')
pi_res = Path('/mnt/2tb_ssd/coding/results')
lap_res = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/results')
# desk_res = Path('/mnt/pishare/results')
desk_res = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/results')
for res_path in [pi_res, lap_res, desk_res]:
    if res_path.exists():
        results_data = res_path
        break
if not results_data:
    note = 'none of the paths for results_data are available'
    print(note)
    pb.push_note(now, note)
