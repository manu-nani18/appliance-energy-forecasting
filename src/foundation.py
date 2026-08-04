"""Part 7 - time-series foundation model (Amazon Chronos).

Chronos is a pretrained transformer that forecasts a series *zero-shot*: no
task-specific training, just feed it the history. We use the small Chronos-Bolt
checkpoint, which runs on CPU and downloads on first use.

The model weights are fetched from the Hugging Face hub the first time this
runs, so an internet connection is needed once. If the package or the weights
are unavailable, we fall back to a seasonal-naive forecast and flag it, so the
pipeline still completes.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from . import config
from .benchmarks import seasonal_naive_forecast
from .utils import score

MODEL_NAME = "amazon/chronos-bolt-small"


def _chronos_forecast(context: np.ndarray, horizon: int):
    """Return (median, low10, high90) using Chronos quantiles."""
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(
        MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(context, dtype=torch.float32)
    quantiles, _ = pipe.predict_quantiles(
        context=ctx, prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0].numpy()  # shape (horizon, 3)
    return q[:, 1], q[:, 0], q[:, 2]


def run(df, horizon=config.HORIZON):
    """Zero-shot 24-hour forecast; falls back gracefully if unavailable."""
    y = df[config.TARGET].astype(float)
    train, test = y.iloc[:-horizon], y.iloc[-horizon:]

    used_fallback = False
    try:
        median, lo, hi = _chronos_forecast(train.values, horizon)
        name = "Chronos"
    except Exception as e:  # no package / no weights / no network
        print(f"[foundation] Chronos unavailable ({e}); using seasonal-naive "
              f"fallback so the pipeline completes.", file=sys.stderr)
        median = seasonal_naive_forecast(train, horizon).values
        lo = median * 0.6
        hi = median * 1.4
        name = "Chronos(fallback=SeasonalNaive)"
        used_fallback = True

    fc = pd.Series(median, index=test.index)
    return {
        "name": name,
        "forecast": fc,
        "lower": pd.Series(lo, index=test.index),
        "upper": pd.Series(hi, index=test.index),
        "metrics": score(test.values, median),
        "train_tail": train, "test": test,
        "used_fallback": used_fallback,
    }
