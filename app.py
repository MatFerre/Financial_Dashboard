import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimizer import optimize_portfolio

st.set_page_config(layout="wide")
st.title("📈 Portfolio Optimization Dashboard")

# Sidebar for user input
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN").split(",")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
method = st.sidebar.selectbox("Optimization Method", ["sharpe", "min_vol", "equal"])

# Fetch historical prices
data = yf.download(tickers, start=start_date, end=end_date)["Close"]
returns = data.pct_change().dropna()

# Optimize weights
weights = optimize_portfolio(returns, method=method)
weights = pd.Series(weights, index=returns.columns)

# Portfolio stats
expected_return = np.dot(weights, returns.mean()) * 252
volatility = np.sqrt(weights.T @ returns.cov() @ weights) * np.sqrt(252)
sharpe_ratio = expected_return / volatility

# Display outputs
st.subheader("Optimized Portfolio Weights")
st.dataframe(weights.apply(lambda x: f"{x:.2%}"))

st.markdown(f"""
**Expected Annual Return:** {expected_return:.2%}  
**Annual Volatility:** {volatility:.2%}  
**Sharpe Ratio:** {sharpe_ratio:.2f}
""")

# Efficient frontier (optional enhancement)
st.subheader("Efficient Frontier")
num_ports = 5000
results = np.zeros((3, num_ports))

for i in range(num_ports):
    w = np.random.dirichlet(np.ones(len(tickers)))
    port_return = np.dot(w, returns.mean()) * 252
    port_vol = np.sqrt(w.T @ returns.cov() @ w) * np.sqrt(252)
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
