import numpy as np
import pandas as pd
import plotly.express as px

FILENAME = "prices_round_0_day_-1.csv"
df = pd.read_csv(f"data/{FILENAME}", delimiter=';')

tomato_mask = df['product'] == 'TOMATOES'
emerald_mask = df['product'] == 'EMERALDS'
tomato_trades = df[tomato_mask]
emerald_trades = df[emerald_mask]

print(tomato_trades['timestamp'])


fig = px.scatter(emerald_trades, x = 'timestamp', y = ['bid_price_1', 'bid_price_2', 'bid_price_3', 'ask_price_1', 'ask_price_2', 'ask_price_3'], color_discrete_sequence=['blue'] * 3 + ['red'] * 3)
#fig = px.scatter(emerald_trades, x = 'timestamp', y = 'bid_price_1')
fig.show()