"""Fast smoke tests on synthetic data - no network required.

Run with:  pytest -q
"""
import numpy as np
import pandas as pd

from src import data, benchmarks, features, config
from src.utils import rmse, score, train_test_split


def _df():
    return data.make_synthetic(n_days=40)


def test_synthetic_shape():
    df = _df()
    assert config.TARGET in df.columns
    assert len(df) == 40 * 24
    assert df.isna().sum().sum() == 0


def test_metrics_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert score(y, y)["MAE"] == 0.0


def test_benchmarks_produce_right_horizon():
    df = _df()
    train, test = train_test_split(df[config.TARGET], horizon=24)
    res = benchmarks.run_all(train, test, horizon=24)
    assert set(res) == {"Mean", "Naive", "SeasonalNaive-daily",
                        "SeasonalNaive-weekly", "Drift"}
    for _name, (fc, m) in res.items():
        assert len(fc) == 24
        assert np.isfinite(m["RMSE"])


def test_seasonal_naive_repeats_last_day():
    df = _df()
    y = df[config.TARGET]
    train, _ = train_test_split(y, horizon=24)
    fc = benchmarks.seasonal_naive_forecast(train, 24, 24)
    assert np.allclose(fc.values, train.iloc[-24:].values)


def test_features_no_leakage_and_dropna():
    df = _df()
    frame = features.build_feature_frame(df)
    assert frame.isna().sum().sum() == 0
    # a lag-24 feature must equal the target 24 rows earlier
    ts = frame.index[100]
    assert np.isclose(frame.loc[ts, f"{config.TARGET}_lag24"],
                      df[config.TARGET].loc[ts - pd.Timedelta(hours=24)])
