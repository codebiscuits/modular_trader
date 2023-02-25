"""this script runs the entire setup scanning process for each timeframe, with 
the appropriate timedelta offset"""

import keys
import time
from binance.client import Client
import binance.exceptions as bx
from pushbullet import Pushbullet
from original_scanner import setup_scan

# import argparse

# setup
if __name__ == '__main__':
    client = Client(keys.bPkey, keys.bSkey)
    pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

    script_start = time.perf_counter()

    print('\n-+-+-+-+-+-+-+-+-+-+-+- Running Setup Scanner -+-+-+-+-+-+-+-+-+-+-+-\n')

    try:
        setup_scan()
    except bx.BinanceAPIException as e:
        print(e)

    script_end = time.perf_counter()

    total_time = script_end - script_start
    print(f"Scanner finished, total time taken: {int(total_time // 60)}m {int(total_time % 60)}s")
