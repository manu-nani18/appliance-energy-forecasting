"""Smoke + correctness tests.

Fast tests run offline on synthetic data. Tests that genuinely need the
Chronos weights (network + torch) are skipped automatically when those are not
installed, so this file always passes in CI but exercises the real model on a
machine that has it.

Run with:  pytest -q
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from src import data, benchmarks, features, config, sarimax_model
from src.utils import rmse, score, train_test_split


def _df(n_days=40):
    return data.make_synthetic(n_days=n_days)


# --------------------------------------------------------------------------
# Data / metrics / features
# --------------------------------------------------------------------------
def test_synthetic_shape():
    df = _df()
    assert config.TARGET in df.columns
    assert len(df) == 40 * 24
    assert df.isna().sum().sum() == 0


def test_metrics_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert score(y, y)["MAE"] == 0.0


def test_features_no_leakage_and_dropna():
    df = _df()
    frame = features.build_feature_frame(df)
    assert frame.isna().sum().sum() == 0
    ts = frame.index[100]
    assert np.isclose(frame.loc[ts, f"{config.TARGET}_lag24"],
                      df[config.TARGET].loc[ts - pd.Timedelta(hours=24)])


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
def test_benchmarks_horizon_and_index_alignment():
    df = _df()
    train, test = train_test_split(df[config.TARGET], horizon=24)
    res = benchmarks.run_all(train, test, horizon=24)
    assert set(res) == {"Mean", "Naive", "SeasonalNaive-daily",
                        "SeasonalNaive-weekly", "Drift"}
    for _name, (fc, m) in res.items():
        assert len(fc) == 24
        assert list(fc.index) == list(test.index)   # aligned to the test window
        assert np.isfinite(m["RMSE"])


# --------------------------------------------------------------------------
# SARIMAX grid: must attempt exactly 147 combinations
# --------------------------------------------------------------------------
def test_sarimax_full_grid_is_147():
    combos = sarimax_model.param_grid(
        config.P_RANGE, config.D_RANGE, config.Q_RANGE)
    assert len(combos) == 147          # 7 * 3 * 7
    assert (0, 0, 0) in combos and (6, 2, 6) in combos


def test_sarimax_grid_records_status_and_failures():
    """Run a tiny non-seasonal grid quickly; check the table schema + that
    failed fits are logged rather than crashing the search."""
    y = _df(20)[config.TARGET]
    train = y.iloc[:-24]
    _order, _aic, table = sarimax_model.grid_search(
        train, seasonal_order=(0, 0, 0, 0),
        p_range=range(0, 2), d_range=range(0, 2), q_range=range(0, 2),
        verbose=False)
    for col in ("p", "d", "q", "seasonal_order", "aic", "status", "error"):
        assert col in table.columns
    assert len(table) == 2 * 2 * 2      # every combination recorded


# --------------------------------------------------------------------------
# Chronos: no silent fallback; real forecast is length 24 and aligned
# --------------------------------------------------------------------------
def _chronos_available():
    return (importlib.util.find_spec("torch") is not None and
            importlib.util.find_spec("chronos") is not None)


def test_chronos_no_silent_fallback():
    """With allow_fallback=False and Chronos unavailable, it must RAISE,
    never return a seasonal-naive result labelled as Chronos."""
    from src import foundation
    df = _df()
    if _chronos_available():
        pytest.skip("Chronos installed; cannot test the unavailable path here")
    with pytest.raises((RuntimeError, ImportError)):
        foundation.run(df, allow_fallback=False)


def test_chronos_fallback_is_labelled_when_opted_in():
    """The fallback must be explicit and clearly labelled, not silent."""
    from src import foundation
    df = _df()
    if _chronos_available():
        pytest.skip("Chronos installed; fallback path not exercised")
    out = foundation.run(df, allow_fallback=True)
    assert out["used_fallback"] is True
    assert "fallback" in out["name"].lower()


@pytest.mark.skipif(not _chronos_available(),
                    reason="torch + chronos-forecasting not installed")
def test_chronos_real_run_length_alignment_and_nonneg():
    from src import foundation
    df = _df()
    out = foundation.run(df, allow_fallback=False)
    test_idx = df[config.TARGET].iloc[-24:].index
    assert out["name"] == "Chronos"
    assert out["used_fallback"] is False
    assert len(out["forecast"]) == 24
    assert list(out["forecast"].index) == list(test_idx)
    assert (out["forecast"].values >= 0).all()
    assert not np.isnan(list(out["metrics"].values())).any()
