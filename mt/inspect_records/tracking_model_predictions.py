import pandas as pd
# from pushbullet import Pushbullet
from pathlib import Path
import trade_records_funcs as trf
from itertools import product
from pprint import pprint
from collections import Counter
from resources.loggers import create_logger
import statistics as stats

from sklearn.metrics import fbeta_score, make_scorer
scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

# pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

logger = create_logger('track_model_preds')

records_folder = Path('/home/ross/coding/pi_2/modular_trader/records')

data = trf.load_all(records_folder, ['closed', 'closed_sim'])

for tf in ['1h', '4h', '12h', '1d']:
    agent = f'trail_fractals_{tf}_None_5_2_1d_volumes_30'
    print('\n', agent)

    l_pred = []
    l_real = []
    s_pred = []
    s_real = []
    for k, d in data[agent].items():
        # print(k)
        # pprint(d)
        if not d['signal'].get('confidence_l'):
            continue
        direction = d['signal']['direction']
        score = d['signal']['score']
        conf_l = d['signal']['confidence_l']
        conf_s = d['signal']['confidence_s']
        conf = conf_l if direction == 'long' else conf_s
        conf_bin = 1 if conf > 0.5 else 0
        rpnl = sum([stage['rpnl'] for stage in d['trade'][1:]])
        rpnl_bin = 1 if rpnl > 0 else 0
        # print(f"{direction}, {conf:.2f}, {rpnl:.2f}")
        if direction == 'long':
            l_pred.append(conf_bin)
            l_real.append(rpnl_bin)
        else:
            s_pred.append(conf_bin)
            s_real.append(rpnl_bin)

    score_l = fbeta_score(l_real[:-20], l_pred[:-20], beta=0.333, zero_division=0)
    score_s = fbeta_score(s_real[:-20], s_pred[:-20], beta=0.333, zero_division=0)

    print(f"score l: {score_l:.1%}, score s: {score_s:.1%}, avg: {stats.mean([score_l, score_s]):.1%}")