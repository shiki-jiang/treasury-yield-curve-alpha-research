# Treasury Yield Curve Modeling and Systematic Signal Research

**Data snapshot:** 2018-01-02 to 2026-10-30 (ETF adjusted prices); 2018-01-02 to 2026-10-30 (U.S. Treasury par yields).

## Yield curve research

Fixed-decay Nelson-Siegel approximation to **par yields**, not zero-coupon prices. Four tenors: 2, 5, 10, and 30 years. Monthly factor VAR(1) forecasts are strictly expanding-window; the no-change forecast is the benchmark.

| Maturity | NS/VAR(1) RMSE (bp) | No-change RMSE (bp) |
|---|---:|---:|
| 2 Yr | 31.65 | 30.03 |
| 5 Yr | 33.05 | 30.62 |
| 10 Yr | 30.25 | 27.86 |
| 30 Yr | 26.49 | 24.53 |

Pre-2021 PCA of monthly yield **changes**: PC1 93.6%, PC2 5.8%, PC3 0.6%.

## Investment research

Signal inputs: previous-month Treasury slope, curvature, and three-month TLT–IEF momentum spread. Expanding ridge regression predicts the next trading-month TLT–IEF return spread. Trading is modeled at the first trading day's closing price; the prior month's signals are available before entry, and the most recent training observation is excluded as an execution-timing safeguard.

Fixed benchmark weights: SHY 40%, IEF 30%, TLT 30%. Strategy holds SHY 40% and varies the IEF/TLT allocation within the other 60%. Assumed cost: 5 basis points per dollar traded. No leverage or shorting.

**Test window:** 2021-01 to 2026-09 (69 completed months).

| Outcome | Signal strategy (net) | Fixed benchmark |
|---|---:|---:|
| Total return | -15.14% | -14.31% |
| Annualized return | -2.81% | -2.65% |
| Annualized volatility | 7.04% | 7.43% |
| Maximum drawdown | -21.47% | -21.20% |

Annualized arithmetic active return: -0.20%; tracking error: 0.84%; information ratio: -0.232.

Factor-adjusted annualized active-return intercept: -0.51%; HAC t-statistic: -1.89; p-value: 0.059. This estimate is not statistically distinguishable from zero at the 5% level.

The strategy is a **research test, not a demonstrated source of alpha**. The benchmark and risk factors are deliberately simple, and findings are sample-dependent.

## Limitations

- Treasury CMT quotes are par yields; Nelson-Siegel fits here are descriptive, not a tradable zero-coupon discount curve.
- Treasury data are dated observations; ETF adjusted closes are vendor-supplied total-return proxies, not executable trades. Trades at the first-day close are an execution approximation.
- Static parameters and signals were chosen in advance of the holdout evaluation in the *code design*, not validated with a separate earlier preregistration; repeated researcher experimentation could introduce data snooping.
- No transaction-impact model, financing, tax, fund-fee changes, ETF premium/discount or borrow constraints; fixed 5-bp one-way transaction cost is assumed.
- The factor-adjusted alpha regression controls two Treasury ETF spreads; a richer risk model could change the estimated intercept and significance.
- Yahoo Finance adjusted price history is subject to revisions. The frozen raw data used here are supplied separately from the public Git repository.

**Research conclusion:** See the forecast benchmark comparison and estimated alpha above. Neither low forecast error nor a positive backtest alone establishes reliable out-of-sample alpha.
