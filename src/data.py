"""Part 1 - data acquisition and preparation.

Downloads the UCI Appliances Energy Prediction dataset (10-minute sampling),
parses the timestamp, checks for missing values and resamples to hourly.

If the machine has no internet access, `make_synthetic()` produces a series
with the same structure (daily + weekly cycles + noise) so the pipeline and
tests can still run end to end.
"""
from __future__ import annotations

import io
import sys
import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download(force: bool = False) -> pd.DataFrame:
    """Fetch the raw 10-minute CSV, caching it under data/raw/."""
    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    import requests
    last_err = None
    for url in config.DATA_MIRRORS:
        try:
            print(f"[data] downloading {url}", file=sys.stderr)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            config.RAW_CSV.write_text(r.text)
            return df
        except Exception as e:  # try the next mirror
            last_err = e
            print(f"[data] failed: {e}", file=sys.stderr)
    raise RuntimeError(f"Could not download dataset. Last error: {last_err}")


# --------------------------------------------------------------------------
# Prepare / resample
# --------------------------------------------------------------------------
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the timestamp and bin the 10-minute data up to hourly values.

    * `Appliances` and `lights` are energy in Wh -> summed over the hour.
    * All sensor / weather channels are instantaneous -> averaged over the hour.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # drop the two random variables shipped for a benchmarking exercise
    df = df.drop(columns=[c for c in ("rv1", "rv2") if c in df.columns])

    energy_cols = [c for c in ("Appliances", "lights") if c in df.columns]
    mean_cols = [c for c in df.columns if c not in energy_cols]

    agg = {c: "sum" for c in energy_cols}
    agg.update({c: "mean" for c in mean_cols})
    hourly = df.resample(config.FREQ).agg(agg)
    return hourly


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Count missing values per column plus any gaps in the hourly index."""
    gaps = pd.date_range(df.index.min(), df.index.max(), freq=config.FREQ)
    n_index_gaps = len(gaps) - len(df)
    report = df.isna().sum()
    report["__index_gaps__"] = n_index_gaps
    return report


def load(force: bool = False, synthetic: bool = False) -> pd.DataFrame:
    """One-call entry point: return the cleaned hourly dataframe."""
    if synthetic:
        return make_synthetic()
    try:
        raw = download(force=force)
        hourly = to_hourly(raw)
        # forward-fill the handful of gaps a resample can introduce
        hourly = hourly.asfreq(config.FREQ).interpolate(limit_direction="both")
        return hourly
    except Exception as e:
        print(f"[data] falling back to synthetic data: {e}", file=sys.stderr)
        return make_synthetic()


# --------------------------------------------------------------------------
# Synthetic fallback (structure mirrors the real series)
# --------------------------------------------------------------------------
def make_synthetic(n_days: int = 137) -> pd.DataFrame:
    """Hourly series with daily + weekly seasonality, useful for tests."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = pd.date_range("2016-01-11", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    daily = 40 * np.sin(2 * np.pi * (t % 24) / 24 - 1.2).clip(min=-0.3)
    weekly = 15 * np.sin(2 * np.pi * (t % 168) / 168)
    base = 90 + daily + weekly
    spikes = rng.random(len(idx)) < 0.05
    noise = rng.gamma(2.0, 15.0, len(idx))
    appliances = np.clip(base + noise + spikes * rng.gamma(3, 60, len(idx)), 10, None)

    temp_out = 6 + 8 * np.sin(2 * np.pi * (t % 24) / 24 - 2.0) + rng.normal(0, 1.5, len(idx))
    rh_out = 80 - 1.5 * temp_out + rng.normal(0, 4, len(idx))
    t1 = 20 + 2 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0, 0.5, len(idx))
    rh1 = 40 + rng.normal(0, 3, len(idx))

    return pd.DataFrame(
        {
            "Appliances": appliances,
            "lights": rng.integers(0, 40, len(idx)).astype(float),
            "T1": t1, "RH_1": rh1,
            "T2": t1 - 1, "RH_2": rh1 + 2,
            "T_out": temp_out, "RH_out": rh_out.clip(1, 100),
            "Press_mm_hg": 755 + rng.normal(0, 5, len(idx)),
            "Windspeed": np.abs(rng.normal(4, 2, len(idx))),
            "Tdewpoint": temp_out - 4 + rng.normal(0, 1, len(idx)),
        },
        index=idx,
    )
