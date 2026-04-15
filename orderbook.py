import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

ROUND = 0
DAY = -2
DATA_PATHNAME = f"/Users/joshuaye/dev/imcprosperity4/imc-p4-dashboard/data/round{ROUND}/"
PRICES_FILENAME = f"prices_round_{ROUND}_day_{DAY}.csv"
TRADES_FILENAME = f"trades_round_{ROUND}_day_{DAY}.csv"

df_prices = pd.read_csv(f"{DATA_PATHNAME}/{PRICES_FILENAME}", delimiter=';')
df_trades = pd.read_csv(f"{DATA_PATHNAME}/{TRADES_FILENAME}", delimiter=';')

PRODUCTS = sorted(df_prices['product'].unique().tolist())

price_to_volume = {
    'bid_price_1': 'bid_volume_1',
    'bid_price_2': 'bid_volume_2',
    'bid_price_3': 'bid_volume_3',
    'ask_price_1': 'ask_volume_1',
    'ask_price_2': 'ask_volume_2',
    'ask_price_3': 'ask_volume_3',
}

SIZE_MAX = 30


def build_figure(product, normalize, qty_range=None, ob_qty_range=None):
    prices = df_prices[df_prices['product'] == product]
    trades = df_trades[df_trades['symbol'] == product].copy().reset_index(drop=True)

    mid = ((prices['bid_price_1'] + prices['ask_price_1']) / 2).values
    timestamp_to_mid = dict(zip(prices['timestamp'].values, mid))

    melted_frames = []
    for price_col, vol_col in price_to_volume.items():
        tmp = prices[['timestamp', price_col, vol_col]].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        tmp['level'] = price_col
        melted_frames.append(tmp)

    plot_df = pd.concat(melted_frames, ignore_index=True)
    plot_df = plot_df.dropna(subset=['price', 'volume'])
    plot_df['side'] = plot_df['level'].str.startswith('bid').map({True: 'bid', False: 'ask'})

    if ob_qty_range is not None:
        plot_df = plot_df[(plot_df['volume'] >= ob_qty_range[0]) & (plot_df['volume'] <= ob_qty_range[1])]

    trades_df = trades[['timestamp', 'price', 'quantity']].copy()
    trades_df.columns = ['timestamp', 'price', 'volume']

    if qty_range is not None:
        trades_df = trades_df[(trades_df['volume'] >= qty_range[0]) & (trades_df['volume'] <= qty_range[1])]

    if normalize:
        plot_df['price'] = plot_df['price'] - plot_df['timestamp'].map(timestamp_to_mid)
        trades_df['price'] = trades_df['price'] - trades_df['timestamp'].map(timestamp_to_mid)

    title = f"{product} ({'Normalized' if normalize else 'Raw'})"
    y_label = 'price - mid' if normalize else 'price'

    fig = px.scatter(plot_df, x='timestamp', y='price', size='volume', color='side',
                     color_discrete_map={'bid': 'blue', 'ask': 'red'},
                     range_x=[0, 10000], size_max=SIZE_MAX,
                     title=title, labels={'price': y_label})

    if not trades_df.empty:
        trade_sizeref = 2 * trades_df['volume'].max() / SIZE_MAX ** 2
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'].tolist(),
            y=trades_df['price'].tolist(),
            mode='markers',
            marker=dict(
                symbol='star',
                color='black',
                size=trades_df['volume'].tolist(),
                sizemode='area',
                sizeref=trade_sizeref,
                sizemin=15,
            ),
            name='trade',
            customdata=trades_df[['volume']].values,
            hovertemplate='price: %{y}<br>quantity: %{customdata[0]}<extra></extra>',
        ))
        fig.data = tuple(t for t in fig.data if t.name != 'trade') + tuple(t for t in fig.data if t.name == 'trade')

    fig.update_traces(selector=dict(name='bid'), marker_opacity=0.5, marker_line_width=0)
    fig.update_traces(selector=dict(name='ask'), marker_opacity=0.5, marker_line_width=0)
    return fig


app = Dash(__name__)

QTY_MIN = int(df_trades['quantity'].min())
QTY_MAX = int(df_trades['quantity'].max())

vol_cols = list(price_to_volume.values())
OB_VOL_MIN = int(df_prices[vol_cols].min().min())
OB_VOL_MAX = int(df_prices[vol_cols].max().max())

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='product-dropdown',
            options=[{'label': p, 'value': p} for p in PRODUCTS],
            value=PRODUCTS[0],
            clearable=False,
            style={'width': '300px'},
        ),
        dcc.RadioItems(
            id='normalize-toggle',
            options=[{'label': 'Raw', 'value': 'raw'}, {'label': 'Normalized', 'value': 'normalized'}],
            value='raw',
            inline=True,
            style={'marginLeft': '20px', 'alignSelf': 'center'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '10px'}),
    html.Div([
        html.Label('Trade quantity filter:', style={'marginRight': '10px'}),
        dcc.RangeSlider(
            id='qty-slider',
            min=QTY_MIN,
            max=QTY_MAX,
            step=1,
            value=[QTY_MIN, QTY_MAX],
            marks={QTY_MIN: str(QTY_MIN), QTY_MAX: str(QTY_MAX)},
            tooltip={'placement': 'bottom', 'always_visible': True},
        ),
    ], style={'padding': '0 20px 10px 20px'}),
    html.Div([
        html.Label('Bid/ask quantity filter:', style={'marginRight': '10px'}),
        dcc.RangeSlider(
            id='ob-qty-slider',
            min=OB_VOL_MIN,
            max=OB_VOL_MAX,
            step=1,
            value=[OB_VOL_MIN, OB_VOL_MAX],
            marks={OB_VOL_MIN: str(OB_VOL_MIN), OB_VOL_MAX: str(OB_VOL_MAX)},
            tooltip={'placement': 'bottom', 'always_visible': True},
        ),
    ], style={'padding': '0 20px 10px 20px'}),
    dcc.Graph(id='orderbook-graph', style={'height': '80vh'}),
])


@app.callback(
    Output('orderbook-graph', 'figure'),
    Input('product-dropdown', 'value'),
    Input('normalize-toggle', 'value'),
    Input('qty-slider', 'value'),
    Input('ob-qty-slider', 'value'),
)
def update_graph(product, normalize, qty_range, ob_qty_range):
    return build_figure(product, normalize == 'normalized', qty_range, ob_qty_range)


if __name__ == '__main__':
    app.run(debug=True)
