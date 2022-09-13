import json
from pprint import pprint
from pathlib import Path

folder = Path('/media/coding/records')
if not folder.exists():
    folder = Path('/home/ross/Documents/backtester_2021/test_records')

filepaths = ['open_trades.json',
             'ot_backup.json',
             'sim_trades.json',
             'st_backup.json',
             'tracked_trades.json',
             'tr_backup.json'
             ]

timeframes = ['30m', '1h', '4h', '12h', '1d']

for tf in timeframes:
    folder_tf = folder / tf
    print(folder_tf)
    for f in folder_tf.glob('*'):
        if f.is_dir():
            for fp in filepaths:
                filepath = f / fp
                if filepath.exists():
                    filepath.unlink()


