import json
from pprint import pprint
from pathlib import Path

folder = Path('/media/coding/market_data')
if not folder.exists():
    folder = Path('test_records')

for f in folder.glob('*'):
    if f.is_dir():
        print(f)
        filepath = f / 'open_trades.json'
        filepath2 = f / 'ot_backup.json'
        if filepath.exists():
            filepath.unlink()
            print(f"deleted {filepath}")
        if filepath2.exists():
            filepath2.unlink()
            print(f"deleted {filepath2}")

