import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from pprint import pprint
import json
from datetime import datetime, timedelta, timezone


def extract_info(trade_record):

    if len(trade_record['trade']) != 2:
        return

    signal = trade_record['signal']
    open = trade_record['trade'][0]
    close = trade_record['trade'][-1]

    return dict(
        pair=signal['pair'],
        sig_trig=signal['trig_price'],
        sig_inval=signal['inval'],
        sig_ratio=signal['inval_ratio'],

        direction=open['direction'],
        open_price=open['exe_price'],
        open_time=datetime.fromtimestamp(float(open['timestamp'])).astimezone(timezone.utc),
        open_stop=open['hard_stop'],

        close_price=close['exe_price'],
        close_time=datetime.fromtimestamp(float(close['timestamp'])).astimezone(timezone.utc),
        close_action=close['action'],
        rpnl=close['rpnl']
    )


def get_ohlc(pair, start, end):
    start_dt = start - timedelta(hours=1, minutes=15)
    end_dt = end + timedelta(hours=1, minutes=15)

    df = pd.read_parquet(f"/home/ross/Documents/backtester_2021/bin_ohlc_5m/{pair}.parquet")

    data = df[(start_dt <= df.timestamp) & (df.timestamp <= end_dt)]

    return data

def plot_trade(agent_name, df, info):
    fig = go.Figure(data=go.Ohlc(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close']))

    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f"st-{lb}-{mult}-up"], mode='lines', name='Supertrend Up'))
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f"st-{lb}-{mult}-dn"], mode='lines', name='Supertrend Down'))

    pair = info['pair']
    direction = info['direction']
    close_type = info['close_action']
    entry_dt = info['open_time']
    entry_price = info['open_price']
    exit_dt = info['close_time']
    exit_price = info['close_price']
    stop = info['sig_inval']

    fig.add_trace(go.Scatter(x=[entry_dt], y=[entry_price], mode='markers',
                             marker=dict(color='blue', size=10), name='Entry'))
    fig.add_trace(go.Scatter(x=[exit_dt], y=[exit_price], mode='markers',
                             marker=dict(color='yellow', size=10), name='Exit'))

    try:
        fig.add_shape(type='line', x0=df.timestamp.iloc[0], x1=df.timestamp.iloc[-1],
            y0=stop, y1=stop, line=dict(color='red', width=2), name='Hard Stop')
    except IndexError:
        print(f"index error for {pair}")
        print(df)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(width=1920, height=1080, autotypenumbers='convert types',
                      title=f"{agent_name} {pair} {close_type} {direction}")
    plot_folder = Path(f"/home/ross/Documents/backtester_2021/trade_plots")
    plot_folder.mkdir(parents=True, exist_ok=True)
    fig.write_image(f"{plot_folder}/{pair}.png")


folder_path = Path('/home/ross/Documents/backtester_2021/records')
for fp in folder_path.glob('*'):
    agent = fp.parts[-1]
    # print(agent)
    records = None
    with open(fp / 'closed_sim_trades.json', 'r') as file:
        try:
            records = json.load(file)
        except json.decoder.JSONDecodeError:
            print('no records yet')
            records = []

    if records:
        # print('\n')
        # print(agent)

        for k, v in records.items():
            info = extract_info(v)

            if info and (info['rpnl'] < -0.99) and (info['close_action'] == 'close'):
                print('')
                # pprint(info)
                df = get_ohlc(info['pair'], info['open_time'], info['close_time'])
                plot_trade(agent, df, info)
                # print(df)
                print(info['pair'], info['rpnl'])
            # elif info:
            #     print(info['pair'], info['rpnl'])
