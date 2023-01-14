"""this script runs the entire setup scanning process for each timeframe, with 
the appropriate timedelta offset"""

import keys
import time
from datetime import datetime
from binance.client import Client
import binance.exceptions as bx
from datetime import timezone
from pushbullet import Pushbullet
from original_scanner import setup_scan
from config import testing

# import argparse

# setup
if __name__ == '__main__':
    client = Client(keys.bPkey, keys.bSkey, testnet=testing)
    pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

    script_start = time.perf_counter()

    print('\n-+-+-+-+-+-+-+-+-+-+-+- Running Setup Scanner -+-+-+-+-+-+-+-+-+-+-+-\n')

    z = [1, 4, 12, 24]
    hour = datetime.now(timezone.utc).hour
    # hour = 0 # for testing all timeframes
    scripts = [tf for tf in z if hour % tf == 0]
    d = {1: '1h', 4: '4h', 12: '12h', 24: '1d'}

    for run_tf in scripts:
        try:
            setup_scan(d[run_tf], None)
        except bx.BinanceAPIException as e:
            print(e)
            continue

    script_end = time.perf_counter()

    total_time = script_end - script_start
    print(f"Scanner finished, total time taken: {int(total_time // 60)}m {int(total_time % 60)}s")
