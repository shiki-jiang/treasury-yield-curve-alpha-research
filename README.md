# Treasury Yield Curve Modeling & Systematic Signal Research

A reproducible Python research project using **observed U.S. Treasury par yields** and **historical dividend-adjusted U.S. Treasury ETF prices**. It tests two distinct questions:

1. Can an expanding-window Nelson–Siegel factor model forecast Treasury yields more accurately than a no-change forecast?
2. Can yield-curve characteristics and past bond ETF returns predict subsequent ETF relative returns and generate benchmark-relative performance after trading costs?

**This is an empirical test, not a claim to have discovered alpha.** No employer-specific material is included.

## Data and sample

| Data | Instruments / series | Dates | Source |
|---|---|---|---|
| U.S. Treasury par yields | 2, 5, 10, 30 years | 2018-01-02 – 2026-10-30 | [U.S. Treasury daily yield curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates) |
| Dividend-adjusted ETF closing prices | SHY, IEF, TLT | 2018-01-02 – 2026-10-30 | Yahoo Finance via [`yfinance`](https://ranaroussi.github.io/yfinance/) |

The enclosed `data/` folder contains **frozen real-data snapshots**, carried forward from a previously verified bond-ETF data collection and official Treasury yield download. They are not simulated. The original ETF data are used locally; `.gitignore` intentionally excludes `data/etf_prices.csv` from public Git tracking. `python run.py --refresh` downloads data directly from their public sources. If cloning from GitHub, run with `--refresh` before the first analysis.

Analysis is frozen to **October 30, 2026** for reproducibility; `--refresh` replaces input files for the **same fixed date range**, not a rolling current sample.

## Methods

**A. Term structure:** Monthly last-available-day Treasury par yields, a three-factor Nelson–Siegel approximation with fixed decay of 3 years, and PCA of pre-2021 monthly yield changes. Expanding multivariate VAR(1) forecasts of the Nelson–Siegel factors predict the following month's four yields. Compare against no-change forecasts with root mean squared error (basis points). With only four observed maturity points and three factors, in-sample curve fitting is near-saturated; the substantive test is **out-of-sample forecasting**. This is a **descriptive fit to par yields** and must not be used to discount actual bond cash flows.

**B. Systematic signal test:** At the beginning of a holding month, use only information from the previous month's last trading day: 10Y–2Y slope, 10Y minus the average of 5Y and 30Y (curve curvature), and trailing 3-month TLT-minus-IEF adjusted-price momentum. An expanding ridge regression predicts the coming month's **TLT–IEF return spread**, with the last training month excluded to remove same-close timing ambiguity. The strategy holds 40% SHY and varies the allocation of the remaining 60% between IEF and TLT. Fixed benchmark: 40% SHY / 30% IEF / 30% TLT. Portfolio entry is approximated at the first trading day's close; returns are measured to the next month's first trading-day close. **No shorting or leverage**; 5-basis-point trading cost per dollar bought or sold.

Training: 2018–2020; holdout: Jan 2021 – Sep 2026 (69 completed trading months). Signals, penalties, allocation rule, and sample split are fixed in the delivered code, not optimized on holdout returns. The model's signals are **not** actual Treasury carry or roll-down estimates; doing that correctly requires consistent spot curves and bond cash-flow assumptions.

## Results from frozen market data

| Outcome | Strategy (net) | Fixed benchmark |
|---|---:|---:|
| Cumulative return | −15.14% | −14.31% |
| Annualized return | −2.81% | −2.65% |
| Annualized volatility | 7.04% | 7.43% |
| Maximum drawdown | −21.47% | −21.20% |

Annualized active return: **−0.20%**; tracking error: **0.84%**; factor-adjusted alpha estimate: **−0.51% per year** (Newey–West/HAC t = −1.89, p = 0.059). **This is not evidence of positive alpha.**

The Nelson–Siegel/AR model's forecast RMSE was **higher** than the simple no-change benchmark at all four tenors. For example, 10Y RMSE was **30.25 bp** versus **27.86 bp** for no-change. Pre-2021 PCA of yield changes assigns **93.6%** of explained total variation to PC1. See [complete research report](reports/research_report.md) for all results and caveats.

## Files

| File | Purpose |
|---|---|
| `research.py` | Curve estimation, PCA, forecasting, signal estimation, walk-forward strategy, HAC statistics |
| `run.py` | Read/fetch and validate data; write report, CSVs, and two PNG charts |
| `data/treasury_yields.csv` | Historical Treasury data snapshot |
| `data/etf_prices.csv` | Local-only, historical adjusted ETF data snapshot; Git-ignored |
| `reports/` | Reproducible research report, forecasts, factors, PCA, backtest, and figures |
| `tests/test_research.py` | Offline deterministic tests, including temporal leakage checks |
| `PROJECT_SPEC.md` | Frozen design and acceptance criteria |

## Mac / PyCharm installation

Open this folder as an independent PyCharm project and select its own `.venv` interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run.py
```

To replace the frozen inputs from official Treasury and Yahoo downloads (requires internet):

```bash
python run.py --refresh
```

The research script **does not place trades**, predict returns for use in a live account, or demonstrate that the signal is profitable. Prices are vendor-provided adjusted-close proxies, not executable quotations. Fixed-model choices and sample results are open to data-snooping concerns, and future data revisions may change the estimates.
