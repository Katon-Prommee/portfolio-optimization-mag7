# Portfolio Optimization (Magnificent 7 + Gold)

This project implements a portfolio optimization strategy using the Magnificent 7 stocks, with an optional allocation to Gold for hedging.

## Objective
Construct a portfolio that minimizes total risk (variance) under the following constraints:
- Each asset must have a minimum weight of 10%
- No asset can exceed 18%
- Fully invested portfolio (weights sum to 100%)

## Assets
- Alphabet (GOOGL)
- Amazon (AMZN)
- Apple (AAPL)
- Tesla (TSLA)
- Meta (META)
- Microsoft (MSFT)
- Nvidia (NVDA)
- Gold (GLD) – optional

## Methodology
- Daily returns are computed from historical price data (Yahoo Finance)
- Covariance matrix is annualized
- Portfolio variance is minimized using constrained optimization (SLSQP)

## Metrics
- Total Return
- CAGR
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- 95% VaR / CVaR
- Skewness / Kurtosis

## Key Insight
This project explores whether adding Gold improves portfolio stability by reducing downside risk and drawdowns.

## How to Run

```bash
pip install -r requirements.txt
python main.py
