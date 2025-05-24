# app_dash.py

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from optimizer import optimize_portfolio
from datetime import date

# Use dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Portfolio Optimizer"

app.layout = dbc.Container([
    html.H2("Portfolio Optimization Dashboard", className="my-4 text-center text-white"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Parameters", className="bg-dark text-light"),
                dbc.CardBody([
                    html.Label("Stock Tickers (comma-separated)", className="text-light"),
                    dcc.Textarea(
                        id='ticker-input',
                        value='AAPL, MSFT, GOOGL, AMZN, TSLA',
                        style={'width': '100%', 'height': 60},
                    ),
                    html.Br(),

                    html.Label("Optimization Method", className="text-light"),
                    dcc.Dropdown(
                        id='method-select',
                        options=[
                            {'label': 'Maximize Sharpe Ratio', 'value': 'sharpe'},
                            {'label': 'Minimize Volatility', 'value': 'min_vol'},
                            {'label': 'Equal Weights', 'value': 'equal'},
                        ],
                        value='sharpe',
                        clearable=False,
                    ),
                    html.Br(),

                    html.Label("Date Range", className="text-white"),
                    dcc.DatePickerRange(
                        id='date-picker',
                        start_date=date(2020, 1, 1),
                        end_date=date.today(),
                        display_format="YYYY-MM-DD"
                    ),
                    html.Br(), html.Br(),

                    dbc.Button("Run Optimization", id="run-btn", color="success", className="d-grid"),
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Portfolio Summary", className="bg-dark text-light"),
                dbc.CardBody([
                    html.Div(id='metrics'),
                    html.H5("Weights", className="text-light"),
                    dash_table.DataTable(
                        id='weights-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left', 'padding': '8px',
                            'backgroundColor': '#2a2a2a', 'color': 'white'
                        },
                        style_header={
                            'backgroundColor': '#444', 'color': 'white',
                            'fontWeight': 'bold'
                        },
                    ),
                ])
            ]),
        ], width=8)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Efficient Frontier", className="bg-dark text-light"),
                dbc.CardBody([
                    dcc.Graph(id='frontier-plot')
                ])
            ])
        ])
    ])
], fluid=True)


@app.callback(
    [Output("metrics", "children"),
     Output("weights-table", "data"),
     Output("weights-table", "columns"),
     Output("frontier-plot", "figure")],
    Input("run-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("method-select", "value"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
)
def update_portfolio(n_clicks, ticker_input, method, start_date, end_date):
    if not ticker_input:
        return "Please enter at least one ticker.", [], [], go.Figure()

    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    if returns.empty:
        return "No return data available for these tickers.", [], [], go.Figure()

    weights = optimize_portfolio(returns, method=method)
    weights = pd.Series(weights, index=returns.columns)

    expected_return = np.dot(weights, returns.mean()) * 252
    volatility = np.sqrt(weights.T @ returns.cov() @ weights) * np.sqrt(252)
    sharpe_ratio = expected_return / volatility

    info = {}
    for ticker in returns.columns:
        try:
            t = yf.Ticker(ticker)
            t_info = t.info
            info[ticker] = {
                "Name": t_info.get("longName", "N/A"),
                "Sector": t_info.get("sector", "N/A")
            }
        except:
            info[ticker] = {"Name": "N/A", "Sector": "N/A"}

    display_df = pd.DataFrame({
        "Company Name": [info[t]["Name"] for t in weights.index],
        "Weight (%)": weights.values * 100,
        "Sector": [info[t]["Sector"] for t in weights.index]
    }).sort_values("Weight (%)", ascending=False)

    display_df["Weight (%)"] = display_df["Weight (%)"].map("{:.2f}%".format)

    columns = [{"name": col, "id": col} for col in display_df.columns]
    data_table = display_df.to_dict("records")

    metrics = dbc.ListGroup([
        dbc.ListGroupItem(f"Expected Annual Return: {expected_return:.2%}", className="bg-dark text-light"),
        dbc.ListGroupItem(f"Annual Volatility: {volatility:.2f}", className="bg-dark text-light"),
        dbc.ListGroupItem(f"Sharpe Ratio: {sharpe_ratio:.2f}", className="bg-dark text-light"),
    ])

    num_ports = 3000
    results = np.zeros((3, num_ports))
    for i in range(num_ports):
        w = np.random.dirichlet(np.ones(len(tickers)))
        port_return = np.dot(w, returns.mean()) * 252
        port_vol = np.sqrt(w.T @ returns.cov().values @ w) * np.sqrt(252)
        port_sharpe = port_return / port_vol
        results[:, i] = [port_vol, port_return, port_sharpe]

    df_results = pd.DataFrame(results.T, columns=["Volatility", "Return", "Sharpe"])
    fig = px.scatter(df_results, x="Volatility", y="Return", color="Sharpe",
                     color_continuous_scale="Turbo", template="plotly_dark", opacity=0.7)
    fig.add_trace(go.Scatter(x=[volatility], y=[expected_return], mode="markers",
                             marker=dict(color="cyan", size=12, symbol="star"),
                             name="Optimized Portfolio"))

    return metrics, data_table, columns, fig


if __name__ == '__main__':
    app.run(debug=True)