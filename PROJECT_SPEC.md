# Project 3: Frozen scope (v1)

**Research objective:** Test whether yield-curve factors forecast next-month U.S. Treasury par yields and whether Treasury curve and momentum signals forecast subsequent Treasury ETF relative returns. This is not assumed to yield positive alpha.

**Real market data:** U.S. Treasury historical daily 2Y, 5Y, 10Y, 30Y par yields; SHY, IEF, TLT historical adjusted closes, January 2018 through October 2026. Fixed evaluation through October 30, 2026; frozen inputs included locally. The public repository does not commit historical Yahoo price files.

**Part A:** Fixed-decay Nelson–Siegel approximation to four par yields (no attempt to price a zero-coupon bond); PCA of pre-2021 monthly yield changes; expanding VAR(1) of NS factors; forecast comparison to no-change for each tenor.

**Part B:** Three features: last-month yield-curve slope, curvature, and lagged 3-month TLT–IEF momentum. Expanding ridge forecast of next one-month TLT–IEF adjusted-price return spread. Train through t-2, enter first trading-day close t, exit first trading-day close t+1. OOS 2021 onward; SHY 40% plus IEF/TLT sleeve 60%, long-only. Benchmark SHY40 / IEF30 / TLT30. Trading cost 5bp per gross traded dollar. Factor-adjusted active-return intercept via HAC(3) regression on IEF–SHY and TLT–IEF ETF spreads.

**Checks:** Reproducible result exports; no look-ahead, exact accounting of gross returns and costs, allocations sum to one, forecast baseline, nonfabricated financial data; no promise of positive alpha. Seven or fewer compact dependencies and at most two source .py scripts plus one tests file.

**Out of scope:** Security-level Treasury returns, tradable bond mispricing, precise carry/roll-down, rate derivatives, transaction-market impact, data-mined strategy search, proof of persistent alpha, and interactive dashboards. Keep those out unless separately requested.
