import time
import itertools
from pathlib import Path
from datetime import datetime
from resources.loggers import create_logger
from trail_fractals_1a import trail_fractals_1a
from trail_fractals_2 import trail_fractals_2

all_start = time.perf_counter()
now = datetime.now().strftime('%Y/%m/%d %H:%M')
print(f"-:--:--:--:--:--:--:--:--:--:-  {now} Running Trail Fractals Fitting  -:--:--:--:--:--:--:--:--:--:-")

running_on_pi = Path('/pi_2.txt').exists()
logger = create_logger('trail_fractals', 'trail_fractals')

width = 5
atr_spacing = 2

fits = list(enumerate(itertools.product(['1h', '4h', '12h', '1d'], ['long', 'short'])))

for n, fit in fits:
    loop_start = time.perf_counter()
    timeframe, side = fit[0], fit[1]
    print(f"\nFit {n+1} of {len(list(fits))}, Fitting {timeframe} {side} models")

    trail_fractals_1a(side, timeframe, width, atr_spacing, 30, '1d_volumes')
    trail_fractals_1a(side, timeframe, width, atr_spacing, 100, '1w_volumes')
    trail_fractals_2(side, timeframe, width, atr_spacing, 0.4)

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"\nThis fit took: {int(loop_elapsed // 3600)}h {int(loop_elapsed // 60) % 60}m {loop_elapsed % 60:.1f}s\n")
    print('-' * 30)

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 3600)}h {int(all_elapsed // 60) % 60}m {all_elapsed % 60:.1f}s")
