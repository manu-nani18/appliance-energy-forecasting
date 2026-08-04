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

```powershell
# Windows / PowerShell (conda base active is fine)
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# If the plain torch wheel is too large, use the CPU index instead:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running

```powershell
python run_pipeline.py                 # FINAL run: real data, full 147-combo SARIMAX, real Chronos
python run_pipeline.py --fast          # smaller SARIMAX grid (quick sanity check ONLY)
python run_pipeline.py --synthetic     # offline demo on synthetic data
python run_pipeline.py --no-foundation # skip Chronos (avoids the model download)
pytest -q                              # run the unit tests
```

The default `python run_pipeline.py` is the submission run: it downloads the
real data, searches the **full** SARIMAX grid, and runs the **real** Chronos
model (no fallback). `--fast` is for sanity checks only and must not be used
for the results you report.

> **Time warning.** The full SARIMAX grid attempts 147 (p, d, q) combinations,
> each with a daily seasonal component (m = 24). On a typical laptop this takes
> roughly **10-40 minutes** depending on CPU. Use `--fast` while developing.

Every figure lands in `outputs/figures/` and every metric table in
`outputs/metrics/` (`model_comparison.csv` is the headline result).

## Foundation model (Chronos)

- Default checkpoint: **`amazon/chronos-t5-small`** (lightweight, CPU-friendly).
- Runs zero-shot; weights download from the Hugging Face hub on first use
  (needs internet once).
- **No silent fallback:** if Chronos cannot run, the pipeline raises a clear
  error explaining why. A seasonal-naive substitute is only used if you pass
  `--allow-chronos-fallback`, and it is then labelled as a fallback.
- The model name and device (cpu/cuda) are recorded in
  `outputs/metrics/chronos_info.csv`.

## Hardware expectations

- CPU-only is fine; no GPU required. Chronos-t5-small runs on CPU in seconds.
- ~1-2 GB free disk for the torch + Chronos install and cached weights.
- The SARIMAX grid is CPU-bound and single-threaded per fit; more cores do not
  speed a single fit but the run is still bounded by the time note above.

## Expected output files

Figures (`outputs/figures/`): `01_full_series.png`, `02_first_week.png`,
`03_seasonal_profiles.png`, `04_decomposition.png`, `05_acf_pacf.png`,
`05b_differencing.png`, `05c_acf_pacf_differenced.png`,
`06_sarimax_residuals.png`, `07_xgb_importance.png`, `08_all_forecasts.png`,
`09_error_diagnostics.png`, `10_sarimax_forecast.png`, `11_xgb_forecast.png`,
`12_chronos_forecast.png`.

Metrics (`outputs/metrics/`): `model_comparison.csv`,
`model_comparison_with_skill.csv`, `sarimax_grid.csv` (147 rows on the full
run), `sarimax_summary.json` / `.csv`, `stationarity_tests.csv`,
`xgb_feature_importance.csv`, `chronos_info.csv`, `summary.json`.

## Troubleshooting

- **`Chronos did not run: PyTorch is not installed`** - `pip install torch`
  (or the CPU index URL above), then re-run.
- **Chronos load fails / no internet** - the weights download once from
  Hugging Face; connect to the internet for the first run.
- **`sarimax_grid.csv` has fewer than 147 rows** - you used `--fast`. Run the
  plain `python run_pipeline.py`.
- **SARIMAX run is very slow** - expected; see the time warning. Use `--fast`
  only for sanity checks.
- **Convergence warnings during the grid** - harmless; failed fits are logged
  in `sarimax_grid.csv` with `status=failed` and the search continues.

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
