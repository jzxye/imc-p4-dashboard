import os
import re
from functools import lru_cache
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

DATA_DIR = "/Users/joshuaye/dev/imcprosperity4/imc-p4-dashboard/data/"
SIZE_MAX = 30

TRADER_COLORS = {
    'Mark 01':   '#e6194b',  # crimson
    'Mark 14':   '#f58231',  # orange
    'Mark 22':   '#ffe119',  # yellow
    'Mark 38':   '#3cb44b',  # green
    'Mark 49':   '#42d4f4',  # cyan
    'Mark 55':   '#911eb4',  # purple
    'Mark 67':   '#f032e6',  # magenta
    'Penelope':  '#7f7f7f',
    'Gary':      '#bcbd22',
    'Gina':      '#17becf',
    'Olga':      '#aec7e8',
}
DEFAULT_TRADER_COLOR = '#000000'

PRICE_VOL_PAIRS = [
    ('bid_price_1', 'bid_volume_1'),
    ('bid_price_2', 'bid_volume_2'),
    ('bid_price_3', 'bid_volume_3'),
    ('ask_price_1', 'ask_volume_1'),
    ('ask_price_2', 'ask_volume_2'),
    ('ask_price_3', 'ask_volume_3'),
]
VOL_COLS = [v for _, v in PRICE_VOL_PAIRS]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_available_rounds():
    return sorted([
        int(d.replace('round', ''))
        for d in os.listdir(DATA_DIR)
        if re.fullmatch(r'round\d+', d)
    ])


def get_available_days(round_num):
    path = f"{DATA_DIR}round{round_num}/"
    days = set()
    for f in os.listdir(path):
        m = re.match(r'prices_round_\d+_day_(-?\d+)\.csv', f)
        if m:
            days.add(int(m.group(1)))
    return sorted(days)


@lru_cache(maxsize=64)
def load_data(round_num, day):
    """Load prices and trades CSVs. Results are cached in memory by (round, day)."""
    path = f"{DATA_DIR}round{round_num}/"
    df_prices = pd.read_csv(
        f"{path}prices_round_{round_num}_day_{day}.csv", delimiter=';'
    )
    trades_path = f"{path}trades_round_{round_num}_day_{day}.csv"
    try:
        df_trades = pd.read_csv(trades_path, delimiter=';')
    except FileNotFoundError:
        df_trades = pd.DataFrame(
            columns=['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity']
        )
    return df_prices, df_trades


def make_trader_options(traders):
    """Checklist options with a colored dot swatch for each trader."""
    return [
        {
            'label': html.Span([
                html.Span('●', style={
                    'color': TRADER_COLORS.get(t, DEFAULT_TRADER_COLOR),
                    'fontSize': '18px',
                    'marginRight': '4px',
                    'verticalAlign': 'middle',
                    'lineHeight': '1',
                }),
                t,
            ]),
            'value': t,
        }
        for t in traders
    ]


def make_solo_options(traders):
    return [{'label': '— all —', 'value': '__all__'}] + [
        {
            'label': html.Span([
                html.Span('●', style={
                    'color': TRADER_COLORS.get(t, DEFAULT_TRADER_COLOR),
                    'fontSize': '14px',
                    'marginRight': '4px',
                }),
                t,
            ]),
            'value': t,
        }
        for t in traders
    ]


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(df_prices, df_trades, product, normalize, qty_range,
                 ob_qty_range, visible_traders, trader_match, ob_display,
                 downsample, uirevision, x_range=None):
    prices = df_prices[df_prices['product'] == product]
    trades = df_trades[df_trades['symbol'] == product].copy().reset_index(drop=True)

    # Mid price for normalisation (always computed from full-resolution data)
    mid = ((prices['bid_price_1'] + prices['ask_price_1']) / 2).values
    timestamp_to_mid = dict(zip(prices['timestamp'].values, mid))

    # ---- Downsample orderbook rows ----
    if downsample > 1:
        prices = prices.iloc[::downsample]

    # ---- Melt orderbook to (timestamp, price, volume, side) ----
    melted = []
    for price_col, vol_col in PRICE_VOL_PAIRS:
        tmp = prices[['timestamp', price_col, vol_col]].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        tmp['side'] = 'bid' if price_col.startswith('bid') else 'ask'
        melted.append(tmp)
    plot_df = pd.concat(melted, ignore_index=True).dropna(subset=['price', 'volume'])

    if ob_qty_range is not None:
        lo, hi = ob_qty_range
        plot_df = plot_df[(plot_df['volume'] >= lo) & (plot_df['volume'] <= hi)]

    # ---- Filter trades ----
    trades_df = trades[['timestamp', 'price', 'quantity', 'buyer', 'seller']].copy()

    if qty_range is not None:
        lo, hi = qty_range
        trades_df = trades_df[(trades_df['quantity'] >= lo) & (trades_df['quantity'] <= hi)]

    has_named = trades_df['buyer'].notna().any() or trades_df['seller'].notna().any()
    if has_named and visible_traders is not None:
        vt = set(visible_traders)
        if trader_match == 'both':
            trades_df = trades_df[
                trades_df['buyer'].isin(vt) &
                trades_df['seller'].isin(vt)
            ]
        else:
            trades_df = trades_df[
                trades_df['buyer'].isin(vt) |
                trades_df['seller'].isin(vt)
            ]

    # ---- Normalise prices ----
    if normalize:
        plot_df = plot_df.copy()
        plot_df['price'] = plot_df['price'] - plot_df['timestamp'].map(timestamp_to_mid)
        trades_df = trades_df.copy()
        trades_df['price'] = trades_df['price'] - trades_df['timestamp'].map(timestamp_to_mid)

    y_label = 'price − mid' if normalize else 'price'
    title = f"{product}  ({'Normalized' if normalize else 'Raw'})"

    # ---- Figure layout: orderbook (72%) + PnL (28%) ----
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        subplot_titles=[title, 'PnL'],
        vertical_spacing=0.07,
    )

    # -- Orderbook: bubbles and/or best bid/ask lines --
    if ob_display in ('bubbles', 'both'):
        ob_sizeref = (2 * plot_df['volume'].max() / SIZE_MAX ** 2) if not plot_df.empty else 1
        for side, rgba in [('bid', 'rgba(31,119,180,0.25)'), ('ask', 'rgba(214,39,40,0.25)')]:
            sub = plot_df[plot_df['side'] == side]
            if sub.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=sub['timestamp'].values,
                y=sub['price'].values,
                mode='markers',
                name=side,
                marker=dict(
                    color=rgba,
                    size=sub['volume'].values,
                    sizemode='area',
                    sizeref=ob_sizeref,
                    sizemin=3,
                    line=dict(width=0),
                ),
                customdata=sub['volume'].values.reshape(-1, 1),
                hovertemplate=(
                    f'ts: %{{x}}<br>{y_label}: %{{y}}<br>vol: %{{customdata[0]}}'
                    '<extra></extra>'
                ),
            ), row=1, col=1)

    if ob_display in ('lines', 'both'):
        for price_col, color, name in [
            ('bid_price_1', 'rgba(31,119,180,0.85)', 'best bid'),
            ('ask_price_1', 'rgba(214,39,40,0.85)',  'best ask'),
        ]:
            line_df = prices[['timestamp', price_col]].dropna()
            if normalize:
                line_df = line_df.copy()
                line_df[price_col] = line_df[price_col] - line_df['timestamp'].map(timestamp_to_mid)
            fig.add_trace(go.Scattergl(
                x=line_df['timestamp'].values,
                y=line_df[price_col].values,
                mode='lines',
                name=name,
                line=dict(color=color, width=1.5),
                hovertemplate=f'ts: %{{x}}<br>{y_label}: %{{y}}<extra></extra>',
            ), row=1, col=1)

    # -- Trade markers (Scatter — fewer points, need triangle/square symbols) --
    if not trades_df.empty:
        trade_sizeref = 2 * trades_df['quantity'].max() / SIZE_MAX ** 2

        if has_named:
            # triangle-up = buyer side, triangle-down = seller side; color = trader
            all_traders = sorted(set(
                trades_df['buyer'].dropna().tolist() +
                trades_df['seller'].dropna().tolist()
            ))
            for trader in all_traders:
                if visible_traders is not None and trader not in visible_traders:
                    continue
                color = TRADER_COLORS.get(trader, DEFAULT_TRADER_COLOR)
                for role, symbol, label_suffix in [
                    ('buyer',  'triangle-up',   'buy'),
                    ('seller', 'triangle-down', 'sell'),
                ]:
                    sub = trades_df[trades_df[role] == trader]
                    if sub.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub['timestamp'].values,
                        y=sub['price'].values,
                        mode='markers',
                        name=f'{trader} ({label_suffix})',
                        legendgroup=trader,
                        marker=dict(
                            symbol=symbol,
                            color=color,
                            opacity=1.0,
                            size=sub['quantity'].values,
                            sizemode='area',
                            sizeref=trade_sizeref,
                            sizemin=8,
                            line=dict(width=1.5, color='rgba(0,0,0,0.6)'),
                        ),
                        customdata=sub[['buyer', 'seller', 'quantity', 'timestamp']].values,
                        hovertemplate=(
                            'ts: %{customdata[3]}<br>'
                            'buyer: %{customdata[0]}<br>'
                            'seller: %{customdata[1]}<br>'
                            f'{y_label}: %{{y}}<br>'
                            'qty: %{customdata[2]}'
                            '<extra></extra>'
                        ),
                    ), row=1, col=1)
        else:
            # Anonymous trades — single square trace
            fig.add_trace(go.Scatter(
                x=trades_df['timestamp'].values,
                y=trades_df['price'].values,
                mode='markers',
                name='trade',
                marker=dict(
                    symbol='square',
                    color='black',
                    opacity=1.0,
                    size=trades_df['quantity'].values,
                    sizemode='area',
                    sizeref=trade_sizeref,
                    sizemin=8,
                    line=dict(width=1.5, color='rgba(255,255,255,0.6)'),
                ),
                customdata=trades_df[['quantity', 'timestamp']].values,
                hovertemplate=f'ts: %{{customdata[1]}}<br>{y_label}: %{{y}}<br>qty: %{{customdata[0]}}<extra></extra>',
            ), row=1, col=1)

    # -- PnL line (Scattergl for large series) --
    pnl_df = prices[['timestamp', 'profit_and_loss']].dropna()
    if not pnl_df.empty:
        fig.add_trace(go.Scattergl(
            x=pnl_df['timestamp'].values,
            y=pnl_df['profit_and_loss'].values,
            mode='lines',
            name='PnL',
            line=dict(color='green', width=1.5),
            hovertemplate='ts: %{x}<br>PnL: %{y:.1f}<extra></extra>',
        ), row=2, col=1)

    ts_max = int(prices['timestamp'].max()) if not prices.empty else 999_900
    x_lo = x_range[0] if x_range is not None else 0
    x_hi = x_range[1] if x_range is not None else ts_max
    fig.update_xaxes(range=[x_lo, x_hi])
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_yaxes(title_text='PnL', row=2, col=1)
    fig.update_layout(
        uirevision=uirevision,
        legend=dict(orientation='v', x=1.01, y=1, tracegroupgap=2),
        margin=dict(r=160),
    )
    return fig


# ---------------------------------------------------------------------------
# App bootstrap — load initial data once
# ---------------------------------------------------------------------------

AVAILABLE_ROUNDS = get_available_rounds()
_r0 = AVAILABLE_ROUNDS[0]
_days0 = get_available_days(_r0)
_d0 = _days0[0]
_prices0, _trades0 = load_data(_r0, _d0)

_DEFAULT_PRODUCTS = sorted(_prices0['product'].unique().tolist())
_DEFAULT_TRADERS = sorted(set(
    _trades0['buyer'].dropna().tolist() + _trades0['seller'].dropna().tolist()
))
_OB_MIN = int(_prices0[VOL_COLS].min().min())
_OB_MAX = int(_prices0[VOL_COLS].max().max())
_QTY_MIN = int(_trades0['quantity'].min()) if not _trades0.empty else 0
_QTY_MAX = int(_trades0['quantity'].max()) if not _trades0.empty else 100

DOWNSAMPLE_OPTIONS = [
    {'label': 'None',  'value': 1},
    {'label': '×2',    'value': 2},
    {'label': '×5',    'value': 5},
    {'label': '×10',   'value': 10},
    {'label': '×20',   'value': 20},
    {'label': '×50',   'value': 50},
]

GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def get_product_groups(products):
    """Return {group_name: [sorted products]} detected from a product list.

    Groups are found by matching all products that share the same first
    underscore-delimited word, then using the longest common prefix as the
    group label (e.g. GALAXY_SOUNDS, SLEEP_POD, UV_VISOR).
    """
    seen = set()
    groups = {}
    for product in products:
        first_word = product.split('_')[0]
        if first_word in seen:
            continue
        seen.add(first_word)
        matching = sorted(p for p in products if p.startswith(first_word + '_'))
        if len(matching) < 2:
            continue
        prefix = os.path.commonprefix(matching).rstrip('_')
        groups[prefix] = matching
    return dict(sorted(groups.items()))


def build_group_figure(df_prices, normalize, downsample, x_range=None):
    products = sorted(df_prices['product'].unique().tolist())
    groups = get_product_groups(products)
    if not groups:
        return go.Figure()

    group_names = list(groups.keys())
    n = len(group_names)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=group_names,
        shared_xaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    y_label = 'mid − start' if normalize else 'mid price'

    for idx, gname in enumerate(group_names):
        row = idx // ncols + 1
        col = idx % ncols + 1
        for pidx, product in enumerate(groups[gname]):
            pdf = df_prices[df_prices['product'] == product]
            if downsample > 1:
                pdf = pdf.iloc[::downsample]
            mid = ((pdf['bid_price_1'] + pdf['ask_price_1']) / 2)
            if normalize:
                first_val = mid.iloc[0] if not mid.empty else 0
                y = (mid - first_val).values
            else:
                y = mid.values
            short_name = product[len(gname) + 1:] if product.startswith(gname + '_') else product
            fig.add_trace(go.Scattergl(
                x=pdf['timestamp'].values,
                y=y,
                mode='lines',
                name=short_name,
                legendgroup=f'pos{pidx}',
                legendgrouptitle_text=f'Variant {pidx + 1}' if idx == 0 else None,
                showlegend=(idx == 0),
                line=dict(color=GROUP_COLORS[pidx % len(GROUP_COLORS)], width=1.5),
                hovertemplate=f'{product}<br>ts: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>',
            ), row=row, col=col)

    if x_range is not None:
        fig.update_xaxes(range=x_range)

    fig.update_yaxes(title_text=y_label, title_font=dict(size=10))
    fig.update_layout(
        height=nrows * 320,
        legend=dict(orientation='v', x=1.01, y=1, tracegroupgap=4),
        margin=dict(r=140, t=40, b=40),
        uirevision='group-overview',
    )
    return fig

app = Dash(__name__)

app.layout = html.Div([

    # ---- Top controls row ----
    html.Div([
        html.Div([
            html.Label('Round', style={'fontSize': '12px'}),
            dcc.Dropdown(
                id='round-dropdown',
                options=[{'label': f'Round {r}', 'value': r} for r in AVAILABLE_ROUNDS],
                value=_r0, clearable=False, style={'width': '120px'},
            ),
        ]),
        html.Div([
            html.Label('Day', style={'fontSize': '12px'}),
            dcc.Dropdown(
                id='day-dropdown',
                options=[{'label': f'Day {d}', 'value': d} for d in _days0],
                value=_d0, clearable=False, style={'width': '100px'},
            ),
        ]),
        html.Div([
            html.Label('Product', style={'fontSize': '12px'}),
            dcc.Dropdown(
                id='product-dropdown',
                options=[{'label': p, 'value': p} for p in _DEFAULT_PRODUCTS],
                value=_DEFAULT_PRODUCTS[0], clearable=False, style={'width': '230px'},
            ),
        ]),
        html.Div([
            html.Label('Price', style={'fontSize': '12px'}),
            dcc.RadioItems(
                id='normalize-toggle',
                options=[
                    {'label': 'Raw',        'value': 'raw'},
                    {'label': 'Normalized', 'value': 'normalized'},
                ],
                value='raw', inline=True,
            ),
        ], style={'alignSelf': 'flex-end', 'paddingBottom': '2px'}),
        html.Div([
            html.Label('Order book', style={'fontSize': '12px'}),
            dcc.RadioItems(
                id='ob-display-radio',
                options=[
                    {'label': 'Bubbles', 'value': 'bubbles'},
                    {'label': 'Lines',   'value': 'lines'},
                    {'label': 'Both',    'value': 'both'},
                ],
                value='bubbles', inline=True,
            ),
        ], style={'alignSelf': 'flex-end', 'paddingBottom': '2px'}),
        html.Div([
            html.Label('Downsample', style={'fontSize': '12px'}),
            dcc.Dropdown(
                id='downsample-dropdown',
                options=DOWNSAMPLE_OPTIONS,
                value=1, clearable=False, style={'width': '90px'},
            ),
        ]),
    ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'flex-end',
              'padding': '10px 20px 0 20px'}),

    # ---- Slider row ----
    html.Div([
        html.Div([
            html.Label('Frame width:', style={'fontSize': '12px'}),
            dcc.Slider(
                id='frame-width-slider',
                min=1_000, max=999_900, step=1_000,
                value=999_900,
                marks={1_000: '1k', 999_900: '999.9k'},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
        ], style={'flex': '2'}),
        html.Div([
            html.Label('Start:', style={'fontSize': '12px'}),
            dcc.Slider(
                id='start-slider',
                min=0, max=999_900, step=100,
                value=0,
                marks={0: '0', 999_900: '999900'},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
        ], style={'flex': '2'}),
        html.Div([
            html.Label('Trade qty:', style={'fontSize': '12px'}),
            dcc.RangeSlider(
                id='qty-slider',
                min=_QTY_MIN, max=_QTY_MAX, step=1,
                value=[_QTY_MIN, _QTY_MAX],
                marks={_QTY_MIN: str(_QTY_MIN), _QTY_MAX: str(_QTY_MAX)},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
        ], style={'flex': '1'}),
        html.Div([
            html.Label('Bid/ask qty:', style={'fontSize': '12px'}),
            dcc.RangeSlider(
                id='ob-qty-slider',
                min=_OB_MIN, max=_OB_MAX, step=1,
                value=[_OB_MIN, _OB_MAX],
                marks={_OB_MIN: str(_OB_MIN), _OB_MAX: str(_OB_MAX)},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'gap': '30px', 'padding': '10px 20px'}),

    # ---- Trader filter ----
    html.Div([
        # Controls row: label + All/None buttons + Solo dropdown
        html.Div([
            html.Label('Traders:', style={
                'fontSize': '12px', 'marginRight': '8px',
                'alignSelf': 'center', 'whiteSpace': 'nowrap',
            }),
            html.Button('All', id='traders-all-btn', n_clicks=0, style={
                'fontSize': '11px', 'padding': '2px 8px', 'cursor': 'pointer',
                'marginRight': '4px', 'borderRadius': '3px',
            }),
            html.Button('None', id='traders-none-btn', n_clicks=0, style={
                'fontSize': '11px', 'padding': '2px 8px', 'cursor': 'pointer',
                'marginRight': '16px', 'borderRadius': '3px',
            }),
            html.Label('Solo:', style={
                'fontSize': '12px', 'marginRight': '6px',
                'alignSelf': 'center', 'whiteSpace': 'nowrap',
            }),
            dcc.Dropdown(
                id='solo-trader-dropdown',
                options=make_solo_options(_DEFAULT_TRADERS),
                value='__all__',
                clearable=False,
                style={'width': '160px', 'fontSize': '12px'},
            ),
            html.Span('│', style={'margin': '0 12px', 'color': '#ccc'}),
            html.Label('Match:', style={
                'fontSize': '12px', 'marginRight': '6px',
                'alignSelf': 'center', 'whiteSpace': 'nowrap',
            }),
            dcc.RadioItems(
                id='trader-match-radio',
                options=[
                    {'label': 'Any party',  'value': 'any'},
                    {'label': 'Both parties', 'value': 'both'},
                ],
                value='any',
                inline=True,
                labelStyle={'marginRight': '10px', 'fontSize': '12px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'}),

        # Color-coded checklist
        dcc.Checklist(
            id='trader-checklist',
            options=make_trader_options(_DEFAULT_TRADERS),
            value=_DEFAULT_TRADERS,
            inline=True,
            labelStyle={'marginRight': '16px', 'fontSize': '13px',
                        'cursor': 'pointer', 'userSelect': 'none'},
        ),
    ], id='trader-filter-div',
       style={'display': 'flex', 'flexDirection': 'column',
              'padding': '0 20px 8px 20px',
              'visibility': 'visible' if _DEFAULT_TRADERS else 'hidden'}),

    # ---- View tabs ----
    dcc.Tabs(id='view-tabs', value='product', style={'padding': '0 20px'}, children=[
        dcc.Tab(label='Product View', value='product', children=[
            dcc.Loading(
                id='graph-loading',
                type='circle',
                children=dcc.Graph(id='orderbook-graph', style={'height': '85vh'}),
            ),
        ]),
        dcc.Tab(label='Group Overview', value='group', children=[
            dcc.Loading(
                id='group-loading',
                type='circle',
                children=dcc.Graph(id='group-overview-graph', style={'minHeight': '160vh'}),
            ),
        ]),
    ]),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output('day-dropdown', 'options'),
    Output('day-dropdown', 'value'),
    Input('round-dropdown', 'value'),
)
def update_days(round_num):
    days = get_available_days(round_num)
    return [{'label': f'Day {d}', 'value': d} for d in days], days[0]


@app.callback(
    Output('product-dropdown', 'options'),
    Output('product-dropdown', 'value'),
    Output('trader-checklist', 'options'),
    Output('trader-checklist', 'value'),
    Output('solo-trader-dropdown', 'options'),
    Output('solo-trader-dropdown', 'value'),
    Output('trader-filter-div', 'style'),
    Output('qty-slider', 'min'),
    Output('qty-slider', 'max'),
    Output('qty-slider', 'value'),
    Output('qty-slider', 'marks'),
    Output('ob-qty-slider', 'min'),
    Output('ob-qty-slider', 'max'),
    Output('ob-qty-slider', 'value'),
    Output('ob-qty-slider', 'marks'),
    Output('frame-width-slider', 'value'),
    Output('start-slider', 'value'),
    Input('round-dropdown', 'value'),
    Input('day-dropdown', 'value'),
)
def update_controls(round_num, day):
    if day is None:
        raise PreventUpdate

    df_prices, df_trades = load_data(round_num, day)
    products = sorted(df_prices['product'].unique().tolist())
    traders = sorted(set(
        df_trades['buyer'].dropna().tolist() + df_trades['seller'].dropna().tolist()
    ))

    qty_min = int(df_trades['quantity'].min()) if not df_trades.empty else 0
    qty_max = int(df_trades['quantity'].max()) if not df_trades.empty else 100

    ob_min = int(df_prices[VOL_COLS].min().min())
    ob_max = int(df_prices[VOL_COLS].max().max())

    ts_max = int(df_prices['timestamp'].max())

    trader_style = {
        'display': 'flex', 'flexDirection': 'column',
        'padding': '0 20px 8px 20px',
        'visibility': 'visible' if traders else 'hidden',
    }

    return (
        [{'label': p, 'value': p} for p in products], products[0],
        make_trader_options(traders), traders,
        make_solo_options(traders), '__all__',
        trader_style,
        qty_min, qty_max, [qty_min, qty_max], {qty_min: str(qty_min), qty_max: str(qty_max)},
        ob_min, ob_max, [ob_min, ob_max], {ob_min: str(ob_min), ob_max: str(ob_max)},
        ts_max,
        0,
    )


@app.callback(
    Output('trader-checklist', 'value', allow_duplicate=True),
    Input('traders-all-btn', 'n_clicks'),
    Input('traders-none-btn', 'n_clicks'),
    Input('solo-trader-dropdown', 'value'),
    State('trader-checklist', 'options'),
    prevent_initial_call=True,
)
def handle_trader_shortcuts(all_clicks, none_clicks, solo, options):
    triggered = callback_context.triggered[0]['prop_id']
    all_values = [o['value'] for o in options]
    if 'all-btn' in triggered:
        return all_values
    if 'none-btn' in triggered:
        return []
    if 'solo-trader' in triggered:
        if solo and solo != '__all__':
            return [solo]
        return all_values
    raise PreventUpdate


@app.callback(
    Output('start-slider', 'max'),
    Output('start-slider', 'marks'),
    Input('frame-width-slider', 'value'),
    Input('round-dropdown', 'value'),
    Input('day-dropdown', 'value'),
)
def update_start_max(frame_width, round_num, day):
    if day is None or frame_width is None:
        raise PreventUpdate
    df_prices, _ = load_data(round_num, day)
    ts_max = int(df_prices['timestamp'].max())
    new_max = max(0, ts_max - frame_width)
    return new_max, {0: '0', new_max: str(new_max)}


@app.callback(
    Output('orderbook-graph', 'figure'),
    Input('round-dropdown', 'value'),
    Input('day-dropdown', 'value'),
    Input('product-dropdown', 'value'),
    Input('normalize-toggle', 'value'),
    Input('qty-slider', 'value'),
    Input('ob-qty-slider', 'value'),
    Input('trader-checklist', 'value'),
    Input('trader-match-radio', 'value'),
    Input('ob-display-radio', 'value'),
    Input('downsample-dropdown', 'value'),
    Input('frame-width-slider', 'value'),
    Input('start-slider', 'value'),
)
def update_graph(round_num, day, product, normalize, qty_range, ob_qty_range,
                 visible_traders, trader_match, ob_display, downsample, frame_width, start):
    if day is None or product is None:
        raise PreventUpdate
    try:
        df_prices, df_trades = load_data(round_num, day)
        if product not in df_prices['product'].values:
            raise PreventUpdate
        x_start = start or 0
        x_end = x_start + (frame_width or 999_900)
        return build_figure(
            df_prices, df_trades, product,
            normalize == 'normalized',
            qty_range, ob_qty_range, visible_traders, trader_match or 'any',
            ob_display or 'bubbles',
            downsample or 1,
            uirevision=f'{round_num}-{day}-{product}',
            x_range=[x_start, x_end],
        )
    except (FileNotFoundError, KeyError):
        raise PreventUpdate


@app.callback(
    Output('group-overview-graph', 'figure'),
    Input('round-dropdown', 'value'),
    Input('day-dropdown', 'value'),
    Input('normalize-toggle', 'value'),
    Input('downsample-dropdown', 'value'),
    Input('frame-width-slider', 'value'),
    Input('start-slider', 'value'),
    Input('view-tabs', 'value'),
)
def update_group_overview(round_num, day, normalize, downsample, frame_width, start, tab):
    if tab != 'group' or day is None:
        raise PreventUpdate
    try:
        df_prices, _ = load_data(round_num, day)
        groups = get_product_groups(sorted(df_prices['product'].unique().tolist()))
        if not groups:
            return go.Figure(layout=go.Layout(
                title='No product groups detected for this round/day.',
                height=400,
            ))
        x_start = start or 0
        x_end = x_start + (frame_width or 999_900)
        return build_group_figure(
            df_prices,
            normalize == 'normalized',
            downsample or 1,
            x_range=[x_start, x_end],
        )
    except (FileNotFoundError, KeyError):
        raise PreventUpdate


if __name__ == '__main__':
    app.run(debug=True)
