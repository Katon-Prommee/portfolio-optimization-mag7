import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# =========================
# CONFIG
# =========================
START_DATE = "2015-01-01"
END_DATE = None
RISK_FREE_RATE = 0.03
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.18
VAR_LEVEL = 0.05

MAG7 = ["GOOGL", "AMZN", "AAPL", "TSLA", "META", "MSFT", "NVDA"]
GOLD = "GLD"

PORTFOLIOS = {
    "No Gold": MAG7,
    "With Gold": MAG7 + [GOLD],
}

PERCENT_METRICS = {
    "Total Return",
    "CAGR",
    "Annualized Return",
    "Annualized Volatility",
    "Max Drawdown",
    "95% VaR (daily)",
    "95% CVaR (daily)",
}

# =========================
# DATA
# =========================
def download_prices(tickers):
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    return prices.dropna()


def compute_returns(prices):
    return prices.pct_change().dropna()

# =========================
# OPTIMIZATION (MIN VAR)
# =========================
def optimize_min_variance(returns):
    tickers = list(returns.columns)
    n = len(tickers)

    if n * MIN_WEIGHT > 1:
        raise ValueError("min_weight infeasible")
    if n * MAX_WEIGHT < 1:
        raise ValueError("max_weight infeasible")

    cov = returns.cov() * 252
    x0 = np.array([1.0 / n] * n)

    bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def objective(w):
        return np.dot(w.T, np.dot(cov, w))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return pd.Series(result.x, index=tickers)

# =========================
# METRICS
# =========================
def max_drawdown(cum):
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()


def var_cvar(returns):
    var = np.quantile(returns, VAR_LEVEL)
    cvar = returns[returns <= var].mean()
    return var, cvar


def evaluate(port_returns):
    port_returns = port_returns.dropna()
    cum = (1 + port_returns).cumprod()

    total_return = cum.iloc[-1] - 1
    years = len(port_returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1

    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)

    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol

    downside = port_returns[port_returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = (ann_return - RISK_FREE_RATE) / downside_vol if downside_vol else np.nan

    mdd = max_drawdown(cum)
    var_95, cvar_95 = var_cvar(port_returns)

    return pd.Series({
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
        "95% VaR (daily)": var_95,
        "95% CVaR (daily)": cvar_95,
        "Skewness": port_returns.skew(),
        "Kurtosis": port_returns.kurtosis(),
    })

# =========================
# FORMAT
# =========================
def format_summary(s):
    s = s.copy()
    for k in PERCENT_METRICS:
        if k in s:
            s[k] *= 100
    return s.round(2)

# =========================
# RUN
# =========================
def run_portfolio(prices, tickers):
    data = prices[tickers].dropna()
    returns = compute_returns(data)

    weights = optimize_min_variance(returns)
    port_returns = returns.dot(weights)

    summary = evaluate(port_returns)

    return weights, summary


def main():
    all_tickers = sorted(set(MAG7 + [GOLD]))
    prices = download_prices(all_tickers)

    results = {}

    for name, tickers in PORTFOLIOS.items():
        weights, summary = run_portfolio(prices, tickers)
        results[name] = (weights, summary)

    for name, (w, s) in results.items():
        print(f"\n{'='*80}")
        print(name)
        print("\nWeights:")
        print((w * 100).round(2).astype(str) + "%")
        print("\nMetrics:")
        print(format_summary(s))

    comp = pd.DataFrame({k: v[1] for k, v in results.items()})
    print(f"\n{'='*80}")
    print("Comparison")
    print(comp.apply(format_summary, axis=0))


if __name__ == "__main__":
    main()