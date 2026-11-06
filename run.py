"""Run Project 3: `python run.py`; use `--refresh` to redownload public inputs."""

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import (ETFS, YIELD_COLS, TENORS, backtest, curve_forecasts,
                      pca_training, performance)

ROOT = Path(__file__).resolve().parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
START, END = "2018-01-01", "2026-11-01"  # Fixed research vintage, end exclusive


def load_data(refresh=False):
    prices_path, yields_path = DATA / "etf_prices.csv", DATA / "treasury_yields.csv"
    if refresh or not (prices_path.exists() and yields_path.exists()):
        import requests
        import yfinance as yf

        raw = yf.download(ETFS, start=START, end=END, auto_adjust=True,
                          group_by="ticker", progress=False, threads=False)
        prices = pd.DataFrame({t: raw[t]["Close"] for t in ETFS})
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        frames = []
        with requests.Session() as session:
            for year in range(2018, 2027):
                url = ("https://home.treasury.gov/resource-center/data-chart-center/"
                       f"interest-rates/daily-treasury-rates.csv/{year}/all?"
                       f"type=daily_treasury_yield_curve&field_tdr_date_value={year}"
                       "&page&_format=csv")
                response = session.get(url, timeout=35)
                response.raise_for_status()
                frame = pd.read_csv(io.StringIO(response.text))
                frames.append(frame[["Date", *YIELD_COLS]])
        yields = pd.concat(frames, ignore_index=True)
        yields["Date"] = pd.to_datetime(yields["Date"])
        yields = yields.set_index("Date").apply(pd.to_numeric, errors="coerce")
        prices, yields = prices.dropna(), yields.dropna()
        DATA.mkdir(exist_ok=True)
        prices.to_csv(prices_path)
        yields.to_csv(yields_path)
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)[ETFS].sort_index()
    yields = pd.read_csv(yields_path, index_col=0, parse_dates=True)[YIELD_COLS].sort_index()
    for name, df in (("ETF prices", prices), ("Treasury yields", yields)):
        df = df.loc[START:pd.Timestamp(END) - pd.Timedelta(days=1)]
        if len(df) < 1500 or df.index.has_duplicates or df.isna().any().any() or (df <= 0).any().any():
            raise ValueError(f"Invalid {name} data: verify data files or rerun with --refresh")
    return (prices.loc[START:pd.Timestamp(END) - pd.Timedelta(days=1)],
            yields.loc[START:pd.Timestamp(END) - pd.Timedelta(days=1)])


def save_report(prices, yields):
    REPORTS.mkdir(exist_ok=True)
    forecast, factors, fitted = curve_forecasts(yields)
    pca, shares = pca_training(yields)
    trial = backtest(prices, yields)
    stats = performance(trial)
    forecast.to_csv(REPORTS / "yield_forecasts.csv", index=False)
    factors.to_csv(REPORTS / "nelson_siegel_factors.csv")
    pca.to_csv(REPORTS / "pca_loadings.csv", index=False)
    trial.to_csv(REPORTS / "strategy_backtest.csv")

    errors = pd.DataFrame.from_dict({tenor: {
        "NS_VAR_RMSE_bp": np.sqrt(np.mean((f.ns_var_pct - f.actual_pct) ** 2)) * 100,
        "No_change_RMSE_bp": np.sqrt(np.mean((f.no_change_pct - f.actual_pct) ** 2)) * 100,
    } for tenor, f in forecast.groupby("tenor", sort=False)}, orient="index")
    errors.to_csv(REPORTS / "forecast_rmse.csv")
    with (REPORTS / "summary.json").open("w") as handle:
        json.dump({**stats, "pca_explained_variance": shares.tolist(),
                   "forecast_rmse_bp": errors.round(4).to_dict(orient="index"),
                   "price_source_start": str(prices.index.min().date()),
                   "price_source_end": str(prices.index.max().date()),
                   "yield_source_start": str(yields.index.min().date()),
                   "yield_source_end": str(yields.index.max().date())}, handle, indent=2)

    last = fitted.index[-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    actual = yields.loc[yields.index.to_period("M") == last, YIELD_COLS].iloc[-1]
    ax.plot(TENORS, actual, "o-", label="Observed Treasury par yields")
    ax.plot(TENORS, fitted.loc[last], "s--", label="Nelson-Siegel approximation")
    ax.set(xlabel="Maturity (years)", ylabel="Yield (%)", title=f"Treasury curve: {last}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "yield_curve.png", dpi=160)
    plt.close(fig)

    wealth = (1 + trial[["net_return", "benchmark_return"]]).cumprod() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    wealth.plot(ax=ax)
    ax.set(xlabel="Holding month", ylabel="Growth of 100", title="Out-of-sample ETF strategy vs benchmark")
    fig.tight_layout()
    fig.savefig(REPORTS / "strategy_performance.png", dpi=160)
    plt.close(fig)

    pct = lambda x: f"{100 * x:.2f}%"
    significance = ("statistically distinguishable from zero at the 5% level"
                    if stats["alpha_hac_p"] < 0.05 else
                    "not statistically distinguishable from zero at the 5% level")
    lines = [
        "# Treasury Yield Curve Modeling and Systematic Signal Research", "",
        f"**Data snapshot:** {prices.index.min().date()} to {prices.index.max().date()} (ETF adjusted prices); "
        f"{yields.index.min().date()} to {yields.index.max().date()} (U.S. Treasury par yields).", "",
        "## Yield curve research", "",
        "Fixed-decay Nelson-Siegel approximation to **par yields**, not zero-coupon prices. "
        "Four tenors: 2, 5, 10, and 30 years. Monthly factor VAR(1) forecasts are strictly expanding-window; "
        "the no-change forecast is the benchmark.", "",
        "| Maturity | NS/VAR(1) RMSE (bp) | No-change RMSE (bp) |", "|---|---:|---:|",
        *[f"| {tenor} | {v['NS_VAR_RMSE_bp']:.2f} | {v['No_change_RMSE_bp']:.2f} |"
          for tenor, v in errors.iterrows()], "",
        "Pre-2021 PCA of monthly yield **changes**: " + ", ".join(f"PC{i+1} {s:.1%}" for i, s in enumerate(shares)) + ".", "",
        "## Investment research", "",
        "Signal inputs: previous-month Treasury slope, curvature, and three-month TLT–IEF momentum spread. "
        "Expanding ridge regression predicts the next trading-month TLT–IEF return spread. "
        "Trading is modeled at the first trading day's closing price; the prior month's signals are available "
        "before entry, and the most recent training observation is excluded as an execution-timing safeguard.", "",
        "Fixed benchmark weights: SHY 40%, IEF 30%, TLT 30%. "
        "Strategy holds SHY 40% and varies the IEF/TLT allocation within the other 60%. "
        "Assumed cost: 5 basis points per dollar traded. No leverage or shorting.", "",
        f"**Test window:** {stats['first_month']} to {stats['last_month']} ({stats['n_months']} completed months).", "",
        "| Outcome | Signal strategy (net) | Fixed benchmark |", "|---|---:|---:|",
        f"| Total return | {pct(stats['strategy']['total_return'])} | {pct(stats['benchmark']['total_return'])} |",
        f"| Annualized return | {pct(stats['strategy']['annual_return'])} | {pct(stats['benchmark']['annual_return'])} |",
        f"| Annualized volatility | {pct(stats['strategy']['annual_volatility'])} | {pct(stats['benchmark']['annual_volatility'])} |",
        f"| Maximum drawdown | {pct(stats['strategy']['max_drawdown'])} | {pct(stats['benchmark']['max_drawdown'])} |",
        "",
        f"Annualized arithmetic active return: {pct(stats['annual_active_return'])}; "
        f"tracking error: {pct(stats['tracking_error'])}; "
        f"information ratio: {stats['information_ratio']:.3f}.", "",
        f"Factor-adjusted annualized active-return intercept: {pct(stats['annual_factor_adjusted_alpha'])}; "
        f"HAC t-statistic: {stats['alpha_hac_t']:.2f}; p-value: {stats['alpha_hac_p']:.3f}. "
        f"This estimate is {significance}.", "",
        "The strategy is a **research test, not a demonstrated source of alpha**. "
        "The benchmark and risk factors are deliberately simple, and findings are sample-dependent.", "",
        "## Limitations", "",
        "- Treasury CMT quotes are par yields; Nelson-Siegel fits here are descriptive, not a tradable zero-coupon discount curve.",
        "- Treasury data are dated observations; ETF adjusted closes are vendor-supplied total-return proxies, "
        "not executable trades. Trades at the first-day close are an execution approximation.",
        "- Static parameters and signals were chosen in advance of the holdout evaluation in the *code design*, "
        "not validated with a separate earlier preregistration; repeated researcher experimentation could introduce data snooping.",
        "- No transaction-impact model, financing, tax, fund-fee changes, ETF premium/discount or borrow constraints; "
        "fixed 5-bp one-way transaction cost is assumed.",
        "- The factor-adjusted alpha regression controls two Treasury ETF spreads; a richer risk model "
        "could change the estimated intercept and significance.",
        "- Yahoo Finance adjusted price history is subject to revisions. "
        "The frozen raw data used here are supplied separately from the public Git repository.", "",
        "**Research conclusion:** See the forecast benchmark comparison and estimated alpha above. "
        "Neither low forecast error nor a positive backtest alone establishes reliable out-of-sample alpha."
    ]
    (REPORTS / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Research completed: {stats['n_months']} out-of-sample trading months. Reports: {REPORTS}")
    print(f"Strategy total return {pct(stats['strategy']['total_return'])}, "
          f"benchmark {pct(stats['benchmark']['total_return'])}, "
          f"factor-adjusted alpha p={stats['alpha_hac_p']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Redownload market data (requires internet)")
    args = parser.parse_args()
    save_report(*load_data(args.refresh))


if __name__ == "__main__":
    main()
