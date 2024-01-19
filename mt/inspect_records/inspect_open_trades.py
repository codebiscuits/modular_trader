from pprint import pprint
import pandas as pd
from pathlib import Path
import json
import trade_records_funcs as trf
from binance.client import Client
import mt.resources.keys as keys
from datetime import datetime, timezone
import mt.resources.utility_funcs as uf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

client = Client(keys.bPkey, keys.bSkey)

records_folder = Path('/home/ross/coding/modular_trader/records')

now = datetime.now().timestamp()

class Position:
    def __init__(self, k, a, b):
        self.agent = k
        self.pair = a
        self.data = b

        self.price = float(client.get_symbol_ticker(symbol=a)['price'])
        self.direction = b['position']['direction']
        self.open_time = uf.scale_number(b['position']['open_time'], 10)
        self.duration = now - self.open_time
        self.current_value = self.price * float(b['position']['base_size'])
        self.liability = float(b['position']['liability'])
        self.pct_of_init_size = float(b['position']['base_size']) / float(b['trade'][0]['base_size'])

        self.exe_price = float(b['trade'][0]['exe_price'])
        self.init_stop = float(b['trade'][0]['hard_stop'])
        self.init_r = (self.exe_price - self.init_stop) / self.exe_price
        if self.direction == 'short':
            self.init_r *= -1

        self.target = b['position'].get('target')

        self.current_stop = float(b['position']['hard_stop'])
        self.open_risk_pct = (((self.price - self.current_stop) / self.price)
                         if self.direction == 'long'
                         else ((self.current_stop - self.price) / self.price))
        self.open_risk_usdt = self.current_value * self.open_risk_pct
        self.open_risk_r = self.open_risk_pct / self.init_r

        # check if it's a new trade (no adds or tps)
        self.new = (b['position']['base_size'] == b['position']['init_base_size']) and (len(b['trade']) == 1)

        if self.new:
            self.pnl_ratio = ((self.price / float(b['trade'][0]['exe_price']))
                         if self.direction == 'long' else
                         (float(b['trade'][0]['exe_price']) / self.price))
        else:
            costs = 0.0
            returns = self.current_value
            for i in b['trade']:
                if i['action'] in ['open', 'add']:
                    costs += float(i['quote_size'])
                elif i['action'] == 'tp':
                    returns += float(i['quote_size'])

            self.pnl_ratio = (returns / costs) if self.direction == 'long' else (costs / returns)

        self.rr = b['signal'].get('rr')
        self.pnl_pct = f"{(self.pnl_ratio - 1):.1%}"
        self.pnl_r = (self.pnl_ratio - 1) / self.init_r
        self.pnl_usdt = float(b['trade'][0]['quote_size']) * (self.pnl_ratio - 1)

    def pos_dict(self):
        return {'agent': self.agent,
             'direction': self.direction,
             'current_value': self.current_value,
             'open_risk_usdt': self.open_risk_usdt,
             'pnl_usdt': self.pnl_usdt,
             'liability': self.liability,
             'duration': self.duration / 3600,
             'pair': self.pair,
             'pnl_pct': self.pnl_pct,
             'pnl_r': self.pnl_r}

def print_stats(pos):
    duration_str = (f"{int(pos.duration // (3600 * 24))}d {int(pos.duration // 3600) % 24}h "
                    f"{int((pos.duration / 60) % 60)}m")
    print('')
    print(f"{pos.pair} {pos.direction}, duration: {duration_str}, {pos.pct_of_init_size:.0%} of initial position. "
          f"Min PnL: {pos.pnl_usdt - pos.open_risk_usdt:.2f} USDT")
    print(f"Value: {pos.current_value:.2f} USDT, liability: {pos.liability:.2f} {pos.data['trade'][0].get('loan_asset')}")
    prog_str = f"({pos.pnl_r / pos.rr:.1%} to {pos.rr:.1f}R target)" if pos.rr else ""
    print(f"PnL: {pos.pnl_pct}, {pos.pnl_usdt:.2f} USDT, {pos.pnl_r:.2f}R {prog_str}")
    print(f"Open risk: {pos.open_risk_pct:.1%}, {pos.open_risk_usdt:.2f} USDT, {pos.open_risk_r:.2f}R")
    alt_exit = f"target: {float(pos.target):.6f}" if pos.rr else f"current stop: {pos.current_stop:.6f}"
    print(f"exe price: {pos.exe_price:.6f}, init stop: {pos.init_stop:.6f}, current_price: {pos.price:.6f}, {alt_exit}")


data = trf.load_all(records_folder, ['open'])

all_positions = []

print('\nopen positions:')
if data:
    for k, v in data.items():
        # print(f"\n{'-' * 20}\n{k}")
        for a, b in v.items():
            pos = Position(k, a, b)
            all_positions.append(pos.pos_dict())
            # print_stats(pos)

pos_df = pd.DataFrame().from_records(all_positions)
print(pos_df)

# Portfolio Summary
print(f"\n\nTotal Exposure: {pos_df.current_value.sum():.2f} USDT, "
      f"Total Open Risk: {pos_df.open_risk_usdt.sum():.2f} USDT, "
      f"Total Borrowed USDT: {pos_df.loc[pos_df.direction == 'long'].liability.sum():.2f} USDT, "
      f"Open PnL: {pos_df.pnl_usdt.sum():.2f} USDT")

# Single plot showing all positions
pos_df['symbol'] = np.where(pos_df.direction == 'long', 'triangle-up', 'triangle-down')
fig = px.scatter(pos_df, x='open_risk_usdt', y='pnl_usdt', size='current_value', color='agent', symbol='symbol')
fig.update_layout(template='plotly_dark')
fig.show()

# # Double plot with longs in one subplot and shorts in the other
# df1 = pos_df.loc[pos_df.direction == 'long']
# df2 = pos_df.loc[pos_df.direction == 'short']
#
# # Creating two subplots
# fig = make_subplots(rows=1, cols=2)
#
# # Adding traces for the first subplot
# for shape in df1['shape1_column'].unique():
#     temp_df = df1[df1['shape1_column'] == shape]
#     fig.add_trace(go.Scatter(
#         x=temp_df['x1_column'],
#         y=temp_df['y1_column'],
#         mode='markers',
#         marker=dict(
#             size=temp_df['size1_column'],
#             color=temp_df['color1_column'],
#             symbol=shape,
#             opacity=0.7,
#             line=dict(width=0.5, color='white')
#         ),
#         name=shape
#     ), row=1, col=1)
#
# # Adding traces for the second subplot
# for shape in df2['shape2_column'].unique():
#     temp_df = df2[df2['shape2_column'] == shape]
#     fig.add_trace(go.Scatter(
#         x=temp_df['x2_column'],
#         y=temp_df['y2_column'],
#         mode='markers',
#         marker=dict(
#             size=temp_df['size2_column'],
#             color=temp_df['color2_column'],
#             symbol=shape,
#             opacity=0.7,
#             line=dict(width=0.5, color='white')
#         ),
#         name=shape
#     ), row=1, col=2)
#
# fig.update_layout(showlegend=True)
#
# fig.show()


# TODO i want a grid of 4 scatter plots, longs at the top, shorts at the bottom, oco agents on the left, trailing stop
#  agents on the right. trailing stop plots should have a 'break-even' line showing where min pnl is positive.

# TODO i could do another chart in the same kind of style but showing each agent as a dot, with all the trades for each
#  agent aggregated together
