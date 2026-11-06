"""Deterministic validation: `python -m pytest -q`."""

import numpy as np
import pandas as pd

from research import (ETFS, YIELD_COLS, backtest, curve_forecasts,
                      make_monthly_panel, ns_factors, ns_loadings,
                      pca_training, performance)


def sample_data():
    dates = pd.bdate_range("2018-01-01", "2026-10-30")
    rng = np.random.default_rng(19)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, (len(dates), 3)), axis=0))
    rates = 3 + np.cumsum(rng.normal(0, 0.012, (len(dates), 1)), axis=0)
    yields = rates + np.array([-0.4, -0.1, 0.1, 0.4])
    return (pd.DataFrame(prices, index=dates, columns=ETFS),
            pd.DataFrame(yields, index=dates, columns=YIELD_COLS))


def test_nelson_siegel_recovers_known_curve():
    beta = np.array([[3.0, -1.0, 2.0], [4.0, 1.0, -0.5]])
    observed = pd.DataFrame(beta @ ns_loadings().T, columns=YIELD_COLS)
    factors, fitted = ns_factors(observed)
    np.testing.assert_allclose(factors.to_numpy(), beta, atol=1e-11)
    np.testing.assert_allclose(fitted.to_numpy(), observed.to_numpy(), atol=1e-11)


def test_forecast_does_not_use_future_month():
    _, yields = sample_data()
    initial, _, _ = curve_forecasts(yields)
    modified = yields.copy()
    modified.loc[modified.index >= "2021-01-01", "10 Yr"] += 2
    altered, _, _ = curve_forecasts(modified)
    initial = initial.loc[initial.month == "2021-01"].reset_index(drop=True)
    altered = altered.loc[altered.month == "2021-01"].reset_index(drop=True)
    np.testing.assert_allclose(initial.ns_var_pct, altered.ns_var_pct)
    np.testing.assert_allclose(initial.no_change_pct, altered.no_change_pct)


def test_pca_trained_only_before_holdout():
    _, yields = sample_data()
    original, share = pca_training(yields)
    yields.loc[yields.index >= "2021-01-01"] *= 1.1
    changed, share_new = pca_training(yields)
    np.testing.assert_allclose(original.iloc[:, 1:].to_numpy(), changed.iloc[:, 1:].to_numpy())
    np.testing.assert_allclose(share, share_new)


def test_signal_dates_precede_holding_month():
    prices, yields = sample_data()
    first = make_monthly_panel(prices, yields)
    modified = yields.copy()
    modified.loc[modified.index >= "2021-01-01", "10 Yr"] += 4
    second = make_monthly_panel(prices, modified)
    np.testing.assert_allclose(first.loc["2021-01", ["slope_bp", "curvature_bp"]],
                               second.loc["2021-01", ["slope_bp", "curvature_bp"]])


def test_no_lookahead_at_first_trade():
    prices, yields = sample_data()
    a = backtest(prices, yields)
    altered = prices.copy()
    altered.loc[altered.index > "2021-01-04", "TLT"] *= 1.3
    b = backtest(altered, yields)
    np.testing.assert_allclose(a.iloc[0][[f"weight_{t}" for t in ETFS]].to_numpy(dtype=float),
                               b.iloc[0][[f"weight_{t}" for t in ETFS]].to_numpy(dtype=float))
    assert a.iloc[0].training_last_month <= "2020-11"


def test_return_accounting_and_constraints():
    trial = backtest(*sample_data())
    weights = trial[[f"weight_{t}" for t in ETFS]]
    assert np.allclose(weights.sum(axis=1), 1)
    assert (weights >= 0).all().all()
    assert np.allclose(trial.gross_return - trial.transaction_cost, trial.net_return)
    assert np.allclose(trial.net_return - trial.benchmark_return, trial.active_return)
    assert np.allclose((weights.to_numpy() * trial[[f"ret_{t}" for t in ETFS]].to_numpy()).sum(axis=1),
                       trial.gross_return)
    assert (trial.turnover >= 0).all()
    summary = performance(trial)
    assert summary["n_months"] == len(trial)
    assert 0 <= summary["alpha_hac_p"] <= 1


def test_drawdown_includes_initial_portfolio_value():
    trial = backtest(*sample_data()).copy()
    trial.loc[trial.index[0], "net_return"] = -0.95
    trial["active_return"] = trial.net_return - trial.benchmark_return
    wealth = (1 + trial.net_return).cumprod()
    expected = (wealth / wealth.cummax().clip(lower=1) - 1).min()
    assert expected < -0.90
    assert np.isclose(performance(trial)["strategy"]["max_drawdown"], expected)
