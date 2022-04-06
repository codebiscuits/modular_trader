import json, keys
from json.decoder import JSONDecodeError
import binance_funcs as funcs
from binance.client import Client
from pathlib import Path
from config import ohlc_data, params
import pandas as pd
from datetime import datetime, timedelta
import statistics as stats
from pushbullet import Pushbullet
import time

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


def set_fixed_risk(strat, market_data, total_bal):
    '''calculates fixed risk setting for new trades based on recent performance 
    and previous setting. if recent performance is very good, fr is increased slightly.
    if not, fr is decreased by thirds'''
    
    def reduce_fr(factor, fr_prev, fr_min, fr_inc):
        '''reduces fixed_risk by factor (with the floor value being fr_min)'''
        ideal = (fr_prev - fr_min) * factor
        reduce = max(ideal, fr_inc)
        return max((fr_prev-reduce), fr_min)
    
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
    with open(f"{market_data}/{strat.name}_bal_history.txt", "r") as file:
        bal_data = file.readlines()
    
    fr_prev = json.loads(bal_data[-1]).get('fr')
    fr_min = params.get('fr_range')[0]
    fr_max = params.get('fr_range')[1]
    fr_inc = (fr_max - fr_min) / 10 # increment fr in 10% steps of the range
    
    bal_0 = total_bal
    bal_1 = json.loads(bal_data[-1]).get('balance')
    bal_2 = json.loads(bal_data[-2]).get('balance')
    bal_3 = json.loads(bal_data[-3]).get('balance')
    bal_4 = json.loads(bal_data[-4]).get('balance')
    
    score = 0
    if bal_0 > bal_1:
        score += 1
    if bal_1 > bal_2:
        score += 0.75
    if bal_2 > bal_3:
        score += 0.5
    if bal_3 > bal_4:
        score += 0.25
    
    if score == 2.5:
        fr = min(fr_prev + (2*fr_inc), fr_max)
    elif score == 2.25:
        fr = min(fr_prev + fr_inc, fr_max)
    elif score >= 1.25:
        fr = fr_prev
    elif score >= 0.75:
        fr = reduce_fr(0.333, fr_prev, fr_min, fr_inc)
    elif score >= 0.5:
        fr = reduce_fr(0.5, fr_prev, fr_min, fr_inc)
    else:
        fr = fr_min
        
    print(f'{fr_prev = }, fr range: {fr_min}-{fr_max}, {bal_0 = }, {bal_1 = }, {bal_2 = }, {bal_3 = }, {bal_4 = }, {score = }')
    if fr != fr_prev:
        note = f'fixed risk adjusted from {round(fr_prev*10000, 1)}bps to {round(fr*10000, 1)}bps'
        pb.push_note(now, note)
    
    print(f'fixed risk perf score: {score}')
    return round(fr, 5)

# def set_ind_r_lim(strat, market_data):
#     '''this function sets the limit for open risk per position. so any open 
#     position which gets to over-extended will trigger the take-profit function. 
    
#     It looks at recent performance to decide if profit-taking needs to be more 
#     or less aggressive'''
    
#     if avg_trade_delta is positive (trades are closing higher than they opened):
#         if avg_tp_delta is negative (trades are closing higher than they took profit):
#             raise indiv_r_limit to reduce tps per trade average
#             this will also restrict tps to only the most extreme profitability
#             so might even raise avg_tp_delta
#         else:
#             lower indiv_r_limit to find equilibrium
#     else:
#         if avg_tp_delta is positive:
#             tp_pct = 100
        

