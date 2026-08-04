# Forecasting Household Appliance Energy Use

Time-series case study on the UCI **Appliances Energy Prediction** dataset.
The project models hourly appliance energy demand with a ladder of models —
from naive benchmarks to a SARIMAX model, a gradient-boosted feature model
(XGBoost) and a pretrained time-series foundation model (Chronos) — and
forecasts the next 24 hours.

## Dataset

- Source: UCI Machine Learning Repository, *Appliances Energy Prediction*
  (`energydata_complete.csv`), ~4.5 months of readings at 10-minute sampling.
- Target: `Appliances` — appliance energy consumption in Wh.
- Covariates: indoor temperatures/humidities from 9 room sensors, outdoor
  weather (temperature, humidity, pressure, wind, dewpoint, visibility) from a
  nearby station, and timestamps.
- The pipeline downloads the file automatically on first run and caches it in
  `data/raw/`. If the download fails, a structurally similar synthetic series
  is generated so the code still runs.

## Repository structure

```
.
├── run_pipeline.py         # runs the whole study end to end
├── requirements.txt
├── src/
│   ├── config.py           # all paths / knobs in one place
│   ├── data.py             # Part 1: download, resample to hourly, missing-value check
│   ├── eda.py              # Part 1: plots, decomposition, ACF/PACF, ADF + KPSS
│   ├── benchmarks.py       # Part 3: mean, naive, daily/weekly seasonal naive, drift
│   ├── sarimax_model.py    # Part 4: AIC grid search, residual diagnostics, CI forecast
│   ├── features.py         # Part 5: sensor/weather/time/lag/rolling covariates
│   ├── ml_model.py         # Part 6: XGBoost, 14-day test, recursive 24h forecast
│   ├── foundation.py       # Part 7: Chronos zero-shot forecast
│   ├── evaluate.py         # Part 8: comparison table, forecast/error plots
│   └── utils.py            # metrics (RMSE/MAE/MAPE/sMAPE), splits, plotting
├── notebooks/analysis.ipynb   # narrative walkthrough
├── tests/test_smoke.py     # fast offline unit tests
└── outputs/
    ├── figures/            # all generated plots
    └── metrics/            # comparison table + per-model CSVs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python run_pipeline.py               # full run: downloads data, all models
python run_pipeline.py --fast        # smaller SARIMAX grid (quick sanity run)
python run_pipeline.py --synthetic   # offline demo on synthetic data
python run_pipeline.py --no-foundation   # skip Chronos (avoids model download)
pytest -q                            # run the unit tests
```

Every figure lands in `outputs/figures/` and every metric table in
`outputs/metrics/` (`model_comparison.csv` is the headline result).

## Method summary

| Part | Model(s) | Idea |
|------|----------|------|
| 1 | EDA | Resample 10-min → hourly; decomposition, ACF/PACF, ADF + KPSS |
| 2 | Problem definition | Target `Appliances`; 24-hour horizon; final 24h as test; RMSE/MAE/MAPE |
| 3 | Benchmarks | Mean, naive, daily (24h) & weekly (168h) seasonal naive, drift |
| 4 | SARIMAX | Daily seasonality (m=24) + outdoor-weather exog; AIC grid p,d,q ∈ [0,6],[0,2],[0,6] |
| 5–6 | XGBoost | Lag, rolling, time-of-day, day-of-week, sensor & weather features |
| 7 | Chronos | Pretrained foundation model, zero-shot |
| 8 | Evaluation | Common metrics, forecast overlays, error diagnostics vs best benchmark |

## Notes on design choices

- **SARIMAX runtime.** A seasonal period of 24 makes each fit expensive, so the
  p,d,q grid is searched with a fixed seasonal order and on a capped window of
  recent history (`config.SARIMAX_TRAIN_DAYS`). Both are documented trade-offs.
- **Conditional vs true forecast.** The SARIMAX and XGBoost 24h forecasts use
  the test window's weather/sensor covariates, which would not truly be known
  at the forecast origin — these are *conditional* forecasts. See the report's
  data-leakage discussion (Part 9, Q5).
- **Reproducibility.** Random seeds are fixed in `config.py`.

## References

See `report/` for the full reference list. Key sources: Candanedo et al.
(2017) *Data-driven prediction models of energy use of appliances in a
low-energy house*; Hyndman & Athanasopoulos, *Forecasting: Principles and
Practice*; Chen & Guestrin (2016) *XGBoost*; Ansari et al. (2024) *Chronos*.
