# Optimizer for portfolio dashboard

import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns, risk_free_rate=0.0, method="sharpe"):
    mu = returns.mean().values  # Expected daily returns
    Sigma = returns.cov().values
    n = len(mu)

    # Initial guess: equally weighted portfolio
    x0 = np.repeat(1/n, n)

    # Constraint: sum of weights = 1
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: long-only (0 ≤ w_i ≤ 1)
    bounds = [(0, 1) for _ in range(n)]

    if method == "equal":
        return x0

    if method == "min_vol":
        # Minimize portfolio standard deviation
        def portfolio_volatility(w):
            return np.sqrt(w.T @ Sigma @ w)
        result = minimize(portfolio_volatility, x0, bounds=bounds, constraints=[cons])
        return result.x

    if method == "sharpe":
        # Maximize Sharpe ratio => minimize negative Sharpe ratio
        def neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w.T @ Sigma @ w)
            return - (port_return - risk_free_rate) / port_vol if port_vol != 0 else np.inf

        result = minimize(neg_sharpe, x0, bounds=bounds, constraints=[cons])
        return result.x

    raise ValueError("Unknown method")
