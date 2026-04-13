import numpy as np
import pandas as pd
import plotly.express as px

ROUND = 0
DAY = -1
DATA_PATHNAME = "/Users/joshuaye/documents/imcprosperity4/imc-p4-dashboard/data"
PRICES_FILENAME = f"prices_round_{ROUND}_day_{DAY}.csv"
TRADES_FILENAME = f"trades_round_{ROUND}_day_{DAY}.csv"

df_prices = pd.read_csv(f"{DATA_PATHNAME}/{PRICES_FILENAME}", delimiter=';')
df_trades = pd.read_csv(f"{DATA_PATHNAME}/{TRADES_FILENAME}", delimiter=';')

tomato_prices_mask = df_prices['product'] == 'TOMATOES'
emerald_prices_mask = df_prices['product'] == 'EMERALDS'
tomato_prices = df_prices[tomato_prices_mask]
emerald_prices = df_prices[emerald_prices_mask]

tomato_trades_mask = df_trades['symbol'] == 'TOMATOES'
emerald_trades_mask = df_trades['symbol'] == 'EMERALDS'
tomato_trades = df_trades[tomato_trades_mask]
emerald_trades = df_trades[emerald_trades_mask]


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
    tmp = emerald_prices[['timestamp', price_col, vol_col]].copy()
    tmp.columns = ['timestamp', 'price', 'volume']
    tmp['level'] = price_col
    melted_frames.append(tmp)

plot_df = pd.concat(melted_frames, ignore_index=True)
plot_df = plot_df.dropna(subset=['price', 'volume'])
plot_df['side'] = plot_df['level'].str.startswith('bid').map({True: 'bid', False: 'ask'})

fig = px.scatter(plot_df, x='timestamp', y='price', size='volume', color='side',
                 color_discrete_map={'bid': 'blue', 'ask': 'red', 'trade': 'black'},
                 range_x=[0, 10000])
#fig = px.scatter(emerald_trades, x = 'timestamp', y = 'bid_price_1')
fig.show()