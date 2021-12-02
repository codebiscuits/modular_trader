from pathlib import Path

not_pairs = ['GBPUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 
             'USDCUSDT', 'PAXUSDT', 'COCOSUSDT', 'SUSDUSDT', 'USDPUSDT', 
             'ADADOWNUSDT', 'LINKDOWNUSDT', 'BNBDOWNUSDT', 'ETHDOWNUSDT']

# ohlc data paths
pi_ohlc_bin = Path('/mnt/2tb_ssd/coding/ohlc_binance_1h')
lap_ohlc_bin = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/ohlc_binance_1h')
desk_ohlc_bin = Path('/mnt/pishare/ohlc_binance_1h')
# desk_ohlc_bin = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/ohlc_binance_1h')
for ohlc_path in [pi_ohlc_bin, lap_ohlc_bin, desk_ohlc_bin]:
    if ohlc_path.exists():
        ohlc_data = ohlc_path
        break
    else:
        print('none of the paths for ohlc_data are available')

# market data paths
pi_md = Path('/mnt/2tb_ssd/coding/market_data')
lap_md = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/market_data')
# desk_md = Path('/mnt/pishare/market_data')
desk_md = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/market_data')
for md_path in [pi_md, lap_md, desk_md]:
    if md_path.exists():
        market_data = md_path
        break

# backtesting results paths
pi_res = Path('/mnt/2tb_ssd/coding/results')
lap_res = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/results')
# desk_res = Path('/mnt/pishare/results')
desk_res = Path('/run/user/1000/gvfs/smb-share:server=raspberrypi.local,share=pishare/results')
for res_path in [pi_res, lap_res, desk_res]:
    if res_path.exists():
        results_data = res_path
        break
