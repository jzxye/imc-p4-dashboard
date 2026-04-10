import numpy as np
import pandas as pd
import plotly.express as px

FILENAME = "prices_round_0_day_-1.csv"
df = pd.read_csv(f"/Users/joshuaye/documents/imcprosperity4/imc-p4-dashboard/data/{FILENAME}", delimiter=';')

tomato_mask = df['product'] == 'TOMATOES'
emerald_mask = df['product'] == 'EMERALDS'
tomato_trades = df[tomato_mask]
emerald_trades = df[emerald_mask]

print(tomato_trades['timestamp'])


price_to_volume = {
    'bid_price_1': 'bid_volume_1',
    'bid_price_2': 'bid_volume_2',
    'bid_price_3': 'bid_volume_3',
    'ask_price_1': 'ask_volume_1',
    'ask_price_2': 'ask_volume_2',
    'ask_price_3': 'ask_volume_3',
}

melted_frames = []
for price_col, vol_col in price_to_volume.items():
    tmp = emerald_trades[['timestamp', price_col, vol_col]].copy()
    tmp.columns = ['timestamp', 'price', 'volume']
    tmp['level'] = price_col
    melted_frames.append(tmp)

plot_df = pd.concat(melted_frames, ignore_index=True)
plot_df = plot_df.dropna(subset=['price', 'volume'])
plot_df['side'] = plot_df['level'].str.startswith('bid').map({True: 'bid', False: 'ask'})

fig = px.scatter(plot_df, x='timestamp', y='price', size='volume', color='side',
                 color_discrete_map={'bid': 'blue', 'ask': 'red'},
                 range_x=[0, 10000])
#fig = px.scatter(emerald_trades, x = 'timestamp', y = 'bid_price_1')
fig.show()