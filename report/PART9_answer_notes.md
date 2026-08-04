# Part 9 — notes to help you answer the six questions

These are talking points tied to what the code computes. Fill in the actual
numbers from `outputs/metrics/model_comparison.csv` after you run the pipeline,
and write the final answers in your own words in the report.

**1. Strongest benchmark and what it says about the structure.**
Look at the RMSE column for Mean / Naive / SeasonalNaive-daily /
SeasonalNaive-weekly / Drift. A seasonal-naive model usually wins, which tells
you appliance use is driven by *repeating daily routine* far more than by a
short-term trend (drift) or the single last value (naive). If daily beats
weekly, the day-to-day rhythm dominates; if weekly beats daily, weekday vs
weekend behaviour matters. Point to the "mean by hour" and "mean by day of
week" profiles (fig 03) as the physical explanation.

**2. Does SARIMAX beat the strongest seasonal benchmark?**
Compare SARIMAX RMSE to the best seasonal-naive RMSE (see the skill-score
column). Discuss: (a) daily seasonality — captured via the m=24 seasonal term;
(b) autocorrelation — check the residual ACF (fig 06): if the bars fall inside
the confidence band and the Ljung-Box p-value > 0.05, the AR/MA structure has
absorbed the autocorrelation; (c) exogenous variables — note whether adding
T_out/RH_out changed the AIC. Appliance spikes are bursty and non-Gaussian, so
a linear SARIMAX often only ties the seasonal benchmark — say so honestly.

**3. Does XGBoost improve when lag / rolling / time / sensor-weather features
are added, and which groups help most?**
Use `outputs/metrics/xgb_feature_importance.csv` and the group-importance
printout. Typically lag and rolling features (recent-past energy) dominate,
time-of-day next, weather/sensor least. Explain why: the strongest predictor of
the next hour's use is the last few hours plus the time of day; indoor/outdoor
climate moves slowly and adds little once lags are present.

**4. Does Chronos beat the benchmark, SARIMAX and XGBoost — and is it worth it?**
Compare RMSE. Chronos is zero-shot (no training), so it is impressive if it
lands near the tuned models, but on a bursty single series it often does *not*
beat a good feature model. Weigh the accuracy delta against the cost: hundreds
of MB of weights, a heavier dependency, slower inference. Argue whether the
complexity is justified for this problem (usually: not clearly).

**5. Which variables are genuinely known at the forecast origin?**
Known: calendar features (hour, day-of-week), and lagged *past* appliance use.
NOT known: the test window's future indoor temperature, humidity and outdoor
weather. Because the SARIMAX and XGBoost 24h forecasts here use the test
window's actual covariates, they are **conditional forecasts**, not true
forecasts. A true operational forecast would need forecasted weather (from a
met service) and would carry that extra uncertainty. This is the key
data-leakage / covariate-availability point — state it clearly.

**6. Which model would you recommend for a real smart home, and why?**
Trade off accuracy, interpretability, uncertainty quantification, compute and
deployment. A strong seasonal benchmark or a lightweight XGBoost with only
lag + calendar features (no future-weather leakage) is usually the pragmatic
recommendation: near-best accuracy, cheap, easy to deploy on an edge device,
and honest about what is known at forecast time. Reserve SARIMAX for its
built-in confidence intervals and Chronos for cold-start situations with no
history to train on.
