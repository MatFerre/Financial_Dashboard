# app.py – Streamlit Dashboard (Basic Version)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimizer import optimize_portfolio

st.set_page_config(layout="wide")
st.title("Portfolio Optimization Dashboard")

# Sidebar inputs
st.sidebar.title("Parameters")
tickers_input = st.sidebar.text_area("Enter one or more tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.sidebar.write("Selected:", ", ".join(tickers))

method_label = st.sidebar.selectbox("Optimization Method", ["Maximize Sharpe Ratio", "Minimize Volatility", "Equal Weights"])
method_map = {
    "Maximize Sharpe Ratio": "sharpe",
    "Minimize Volatility": "min_vol",
    "Equal Weights": "equal"
}
method = method_map[method_label]

start = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2020-01-01")
end = st.sidebar.text_input("End Date (YYYY-MM-DD)", str(pd.to_datetime("today").date()))

try:
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    if start_date > end_date:
        st.sidebar.error("Start must be before end date.")
except Exception:
    st.sidebar.error("Invalid date format.")
    st.stop()

# Download data
data = yf.download(tickers, start=start_date, end=end_date)["Close"]
returns = data.pct_change().dropna()

if returns.empty:
    st.warning("No returns data available. Please check the tickers or date range.")
    st.stop()

# Optimization
weights = optimize_portfolio(returns, method=method)
weights = pd.Series(weights, index=returns.columns)

# Portfolio stats
expected_return = np.dot(weights, returns.mean()) * 252
volatility = np.sqrt(weights.T @ returns.cov() @ weights) * np.sqrt(252)
sharpe_ratio = expected_return / volatility

# Get additional info: sector and long name
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

# Display results
st.subheader("Optimized Portfolio Weights")
display_df = pd.DataFrame({
    "Company Name": [info[t]["Name"] for t in weights.index],
    "Weight": weights,
    "Sector": [info[t]["Sector"] for t in weights.index]
})

display_df_sorted = display_df.sort_values("Weight", ascending=False)
st.dataframe(display_df_sorted.style.format({"Weight": "{:.2%}"}))

# Display metrics
st.subheader("Optimized Portfolio Metrics")
st.markdown(f"**Expected Annual Return:** {expected_return:.2%}")
st.markdown(f"**Annual Volatility:** {volatility:.2%}")
st.markdown(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

# Efficient frontier (sample portfolios)
st.subheader("Efficient Frontier (Simulated)")
num_ports = 5000
results = np.zeros((3, num_ports))

for i in range(num_ports):
    w = np.random.dirichlet(np.ones(len(tickers)))
    port_return = np.dot(w, returns.mean()) * 252
    port_vol = np.sqrt(w.T @ returns.cov().values @ w) * np.sqrt(252)
    port_sharpe = port_return / port_vol
    results[:, i] = [port_vol, port_return, port_sharpe]

df_results = pd.DataFrame(results.T, columns=["Volatility", "Return", "Sharpe"])

fig, ax = plt.subplots()
sc = ax.scatter(df_results["Volatility"], df_results["Return"], c=df_results["Sharpe"], cmap="viridis", alpha=0.7)
ax.scatter(volatility, expected_return, color="red", marker="*", s=100, label="Optimized")
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected Return")
ax.legend()
st.pyplot(fig)