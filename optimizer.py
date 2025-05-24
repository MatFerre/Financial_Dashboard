import numpy as np
import cvxpy as cp

def optimize_portfolio(returns, risk_free_rate=0.0, method="sharpe"):
    mu = returns.mean().values
    Sigma = 0.5 * (returns.cov().values + returns.cov().values.T)  # ensure symmetry
    n = len(mu)

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    portfolio_return = mu @ w
    portfolio_variance = cp.quad_form(w, Sigma)

    if method == "sharpe":
        target_vol = cp.Parameter(nonneg=True, value=1.0)
        objective = cp.Maximize(portfolio_return - risk_free_rate)
        constraints.append(portfolio_variance <= target_vol**2)
    elif method == "min_vol":
        objective = cp.Minimize(portfolio_variance)
    elif method == "equal":
        return np.repeat(1 / n, n)
    else:
        raise ValueError("Unknown optimization method")

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value
