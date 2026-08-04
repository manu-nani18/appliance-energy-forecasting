# Report outline (target 6–8 pages, references ≤ 0.5 page)

Write this yourself — this is a scaffold, not the report. Drop in the figures
from `outputs/figures/` and the numbers from `outputs/metrics/`.

1. **Introduction (0.5 p)** — problem, why short-horizon appliance forecasting
   matters (demand response, smart homes), what you set out to compare.

2. **Data & preprocessing (1 p)** — dataset, 10-min → hourly resampling (sum
   energy, mean sensors), missing-value check. Figs 01–02.

3. **EDA & stationarity (1 p)** — daily/weekly seasonality (fig 03),
   decomposition (fig 04), ACF/PACF (fig 05), ADF + KPSS table
   (`stationarity_tests.csv`). State the conclusion: seasonal, and how you made
   it stationary (differencing).

4. **Forecasting problem (0.5 p)** — target, 24h horizon, train/test split,
   metrics (RMSE primary; MAE, MAPE, sMAPE secondary).

5. **Models (1.5 p)** — one short paragraph each: benchmarks, SARIMAX (grid
   search + residual diagnostics, fig 06), feature engineering + XGBoost (fig
   07), Chronos. Say *why* each choice, not just what.

6. **Results (1.5 p)** — the comparison table (`model_comparison.csv`), the
   all-forecasts plot (fig 08), error diagnostics (fig 09), per-model forecast
   plots (figs 10–12). **Critical analysis**: explain *why* results came out as
   they did, and compare forecasts to the real values held out.

7. **Answers to the six questions (0.5–1 p)** — see PART9_answer_notes.md.

8. **Limitations & future work (0.5 p)** — conditional vs true forecast, single
   house, bursty target, no probabilistic scoring for XGBoost; future: multi-
   horizon, quantile loss, weather-forecast inputs, per-appliance modelling.

9. **References (≤ 0.5 p)** — Candanedo et al. 2017; Hyndman & Athanasopoulos
   2021; Chen & Guestrin 2016; Ansari et al. 2024 (Chronos); statsmodels.
