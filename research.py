"""Yield-curve estimation, monthly forecasts, and an ETF signal backtest."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

TENORS = np.array([2.0, 5.0, 10.0, 30.0])
YIELD_COLS = ["2 Yr", "5 Yr", "10 Yr", "30 Yr"]
ETFS = ["SHY", "IEF", "TLT"]
FEATURES = ["slope_bp", "curvature_bp", "momentum_spread"]
START_TEST = pd.Period("2021-01", freq="M")
BENCHMARK = np.array([0.40, 0.30, 0.30])


def monthly(frame):
    """Last available trading-day observation in each calendar month."""
    return frame.groupby(frame.index.to_period("M")).last()


def ns_loadings(t=TENORS, decay=3.0):
    """Fixed-decay Nelson-Siegel loadings (maturities in years)."""
    x = np.asarray(t, dtype=float) / decay
    slope = -np.expm1(-x) / x
    return np.column_stack((np.ones_like(x), slope, slope - np.exp(-x)))


def ns_factors(yields):
    """Approximate four Treasury *par* yields; do not interpret as spot rates."""
    matrix = ns_loadings()
    values = yields[YIELD_COLS].to_numpy(dtype=float)
    factors = values @ np.linalg.pinv(matrix).T
    fitted = factors @ matrix.T
    names = ["level", "slope", "curvature"]
    return (pd.DataFrame(factors, index=yields.index, columns=names),
            pd.DataFrame(fitted, index=yields.index, columns=YIELD_COLS))


def curve_forecasts(yields):
    """Expanding VAR(1) forecasts of Nelson-Siegel factors vs no-change baseline."""
    observed = monthly(yields)[YIELD_COLS]
    factors, fitted = ns_factors(observed)
    b = factors.to_numpy()
    rows = []
    for i in range(25, len(observed) - 1):
        target_month = observed.index[i + 1]
        if target_month < START_TEST:
            continue
        history = b[:i + 1]
        coefficients = np.linalg.lstsq(
            np.c_[np.ones(i), history[:-1]], history[1:], rcond=None
        )[0]
        predicted = (np.r_[1.0, b[i]] @ coefficients) @ ns_loadings().T
        for j, tenor in enumerate(YIELD_COLS):
            rows.append({"month": str(target_month), "tenor": tenor,
                         "actual_pct": observed.iloc[i + 1, j],
                         "ns_var_pct": predicted[j], "no_change_pct": observed.iloc[i, j]})
    return pd.DataFrame(rows), factors, fitted


def pca_training(yields):
    """PCA on pre-2021 monthly changes in four observed par yields."""
    changes = monthly(yields)[YIELD_COLS].diff().dropna()
    train = changes.loc[changes.index < START_TEST].to_numpy()
    centered = train - train.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return pd.DataFrame({"tenor": YIELD_COLS,
                         **{f"PC{i + 1}": vt[i] for i in range(3)}}), s[:3] ** 2 / (s ** 2).sum()


def make_monthly_panel(prices, yields):
    """Trade from first close of month t to first close of t+1.

    Signals for t are calculated at the end of month t-1, before entry.
    """
    beginning = prices[ETFS].groupby(prices.index.to_period("M")).first()
    ending = monthly(prices[ETFS])
    rates = monthly(yields)
    signals = pd.DataFrame(index=rates.index)
    signals["slope_bp"] = 100 * (rates["10 Yr"] - rates["2 Yr"])
    signals["curvature_bp"] = 100 * (rates["10 Yr"] - (rates["5 Yr"] + rates["30 Yr"]) / 2)
    signals["momentum_spread"] = ending["TLT"].pct_change(3) - ending["IEF"].pct_change(3)
    returns = beginning.shift(-1).div(beginning).sub(1).add_prefix("ret_")
    return signals.shift(1).join(returns).dropna()


def predict_spread(history, current, penalty=8.0):
    """Fixed-penalty ridge regression, scaled using training rows only."""
    x = history[FEATURES].to_numpy()
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    x = (x - mu) / sd
    y = (history["ret_TLT"] - history["ret_IEF"]).to_numpy()
    coefficient = np.linalg.solve(x.T @ x + penalty * np.eye(len(FEATURES)),
                                  x.T @ (y - y.mean()))
    return float(y.mean() + ((current[FEATURES].to_numpy(dtype=float) - mu) / sd) @ coefficient)


def backtest(prices, yields, start=START_TEST):
    """Walk-forward strategy; excludes the most recent completed month from training."""
    panel = make_monthly_panel(prices, yields)
    holdings, records = BENCHMARK.copy(), []
    for i, (month, row) in enumerate(panel.iterrows()):
        if month < start:
            continue
        history = panel.iloc[:i - 1]  # t-2 or earlier: no same-close execution assumption
        if len(history) < 24:
            raise ValueError("Need at least 24 fully observed training months")
        prediction = predict_spread(history, row)
        long_share = np.clip(0.5 + prediction / 0.03, 0.20, 0.80)
        weight = np.array([0.40, 0.60 * (1 - long_share), 0.60 * long_share])
        r = row[[f"ret_{t}" for t in ETFS]].to_numpy(dtype=float)
        turnover = float(np.abs(weight - holdings).sum())
        cost = turnover * 0.0005  # 5 basis points per dollar traded
        gross = float(weight @ r)
        bench = float(BENCHMARK @ r)
        records.append({"month": str(month), "prediction": prediction,
                        "realized_spread": r[2] - r[1],
                        "training_last_month": str(history.index[-1]),
                        **{f"weight_{t}": weight[k] for k, t in enumerate(ETFS)},
                        **{f"ret_{t}": r[k] for k, t in enumerate(ETFS)},
                        "gross_return": gross, "net_return": gross - cost,
                        "benchmark_return": bench, "active_return": gross - cost - bench,
                        "turnover": turnover, "transaction_cost": cost})
        holdings = weight * (1 + r) / (1 + gross)  # portfolio drift between trades
    return pd.DataFrame(records).set_index("month")


def performance(backtest_monthly):
    """Net performance, benchmark-relative risk, and HAC factor-adjusted alpha."""
    m = backtest_monthly

    def metrics(col):
        x = m[col]
        growth = (1 + x).cumprod()
        return {"total_return": float(growth.iloc[-1] - 1),
                "annual_return": float(growth.iloc[-1] ** (12 / len(x)) - 1),
                "annual_volatility": float(x.std(ddof=1) * np.sqrt(12)),
                "max_drawdown": float((growth / growth.cummax().clip(lower=1) - 1).min())}

    factors = sm.add_constant(pd.DataFrame({
        "intermediate_exposure": m["ret_IEF"] - m["ret_SHY"],
        "long_duration_exposure": m["ret_TLT"] - m["ret_IEF"],
    }))
    fit = sm.OLS(m["active_return"], factors).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    active = m["active_return"]
    te = float(active.std(ddof=1) * np.sqrt(12))
    return {"n_months": len(m), "first_month": str(m.index[0]),
            "last_month": str(m.index[-1]),
            "strategy": metrics("net_return"), "benchmark": metrics("benchmark_return"),
            "annual_active_return": float(active.mean() * 12),
            "tracking_error": te,
            "information_ratio": float(active.mean() * np.sqrt(12) / active.std(ddof=1)),
            "turnover_sum": float(m["turnover"].sum()),
            "cost_total_return_points": float(m["transaction_cost"].sum()),
            "annual_factor_adjusted_alpha": float(fit.params["const"] * 12),
            "alpha_hac_t": float(fit.tvalues["const"]),
            "alpha_hac_p": float(fit.pvalues["const"])}
