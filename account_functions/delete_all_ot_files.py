import json
from pprint import pprint
from pathlib import Path

folder = Path('/media/coding/market_data')
if not folder.exists():
    folder = Path('/home/ross/Documents/backtester_2021/test_records')

filepaths = ['open_trades.json',
             'ot_backup.json',
             'sim_trades.json',
             'st_backup.json',
             'tracked_trades.json',
             'tr_backup.json'
             ]

for f in folder.glob('*'):
    if f.is_dir():
        for fp in filepaths:
            filepath = f / fp
            if filepath.exists():
                filepath.unlink()


