# Can the forecast be improved? (data v2, all runs logged in MLflow)

Produced by `experiments/run_all.py` (about 5 minutes) on corrected data (see `DATA_CORRECTION.md`; an earlier version of this
study ran on data with swapped Sprint Qualifying sessions and is superseded). Runs: MLflow experiment `f1-weekend-forecasting`,
tag `data_version=v2-session-names` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`; the store is gitignored).

## Protocol

* Walk-forward: a forecast for weekend *t* sees only weekends before *t* (tests scramble the target weekend and everything after
  it and check the predictions do not move). Practice features only use sessions that start before the team lock (start of
  Qualifying, or of the Sprint on Sprint weekends; see `docs/OPENF1_NOTES.md`).
* **dev = 2024** (selection), **val = 2025** (confirmation), **test = 2026** (held out; a pre-registration run fixes the candidate and
  the decision rule before scoring).
* Differences are paired bootstraps over weekends (5000 resamples), "CI" = 95% interval of the MAE change. Weekends are treated as
  independent; each split has 24 weekends (test: 17), so effects under about 0.2 MAE are hard to establish.
* Metric: driver fantasy points per weekend, no Driver of the Day, as in the earlier backtest.

## Results (driver MAE, lower is better)

| Forecaster | dev 2024 | val 2025 | change vs simulator, dev [CI] | change vs simulator, val [CI] |
|---|---|---|---|---|
| last-1 / last-3 / last-5 mean | 10.70 / 8.99 / 8.84 | 13.26 / 10.77 / 10.16 | | |
| season-to-date mean | 8.50 | 10.12 | | |
| EWMA (alpha 0.2), strongest naive baseline | 8.30 | 9.80 | +0.25 [-0.09, +0.59] | +0.44 [+0.09, +0.83] |
| Simulator, defaults | 8.05 | 9.36 | | |
| Gradient-boosted regression (depth 2, 100 trees, lr 0.03) | 8.02 | 9.25 | -0.04 [-0.40, +0.34] | -0.11 [-0.34, +0.14] |
| **Simulator + GBM, adaptive weights (NNLS on earlier weekends)** | 7.92 | 9.20 | -0.14 [-0.29, +0.04] | -0.16 [-0.26, -0.04] |
| Simulator 40% + GBM 60% (weight picked on 2024) | 7.90 | 9.17 | -0.16 [-0.37, +0.08] | -0.18 [-0.34, -0.02] |

GBM settings were chosen on 2024 only. Constructors (shift the simulator's constructor forecast by the summed driver-level
correction): MAE change -0.56 [-0.85, -0.28] (2024) and -0.46 [-0.77, -0.16] (2025) for the adaptive ensemble; 40/60 blend -0.71 / -0.59.

**Reading:** the ensemble is better than the simulator in both years, with the interval excluding 0 on 2025 but, after the data
correction, no longer on 2024. The gain is about 0.15 points per driver-weekend (1.5-2%).

What did **not** help:

* Time-varying team strength (`team_block`), DNF settings, and 96 random draws over 12 simulator settings: every variant is within
  about 0.02 MAE of the default on both years (best of 96 on 2024: 8.00 vs 8.07; the five best re-run on 2025 changed MAE by -0.04
  to +0.04, every CI including 0).
* Adding EWMA to the ensemble (7.88 / 9.16, no better than two models); feeding the simulator's forecast into the GBM (no better than the GBM alone).

## Practice sessions and Sprint Qualifying (known at the team lock)

Practice-pace rank correlates about +0.5 with the weekend's points rank, but most of that is already in the form features.

| Candidate | dev MAE | val MAE | vs previous-best ensemble, dev [CI] | val [CI] |
|---|---|---|---|---|
| Simulator + known Sprint Qualifying (Sprint weekends); compared with the plain simulator, not the ensemble | 8.08 | 9.30 | +0.03 [0.00, +0.06] | -0.06 [-0.16, +0.02] |
| GBM + practice features | 8.00 | 9.27 | | |
| NNLS(simulator, GBM + practice + SQ) (best on dev) | 7.89 | 9.18 | -0.03 [-0.09, +0.02] | -0.02 [-0.11, +0.08] |
| NNLS(simulator + known SQ, GBM + practice) (best on val, picked after seeing val) | 7.90 | 9.13 | -0.02 [-0.08, +0.04] | -0.07 [-0.14, -0.005] |
| Ridge correction of the simulator on practice features | 8.12-8.21 | 9.22-9.27 | worse | no better than the ensemble |

**Not confirmed.** The practice variants move the same way (about 0.02-0.07 better) but no interval excludes 0 for the variant
chosen on dev, and the one that does on 2025 was picked by looking at 2025. They were therefore **not** taken to the 2026 holdout.

## Held-out 2026 (17 weekends; scored twice, the second time only because of the data correction)

| | driver MAE | change vs simulator [CI] | constructor MAE | change vs simulator [CI] |
|---|---|---|---|---|
| Simulator, defaults | 11.02 | | 15.95 | |
| Simulator + GBM, adaptive (pre-registered primary) | 10.84 | -0.18 [-0.34, +0.00] | 16.29 | +0.34 [-0.00, +0.72] |
| 40/60 blend | 10.81 | -0.20 [-0.46, +0.08] | 16.51 | +0.56 [+0.00, +1.17] |
| GBM alone | 10.91 | -0.11 [-0.51, +0.33] | | |
| EWMA 0.2 / last-3 | 11.21 / 12.20 | +0.19 / +1.18 | | |

By the rule fixed beforehand (CI must exclude 0) **the driver improvement is not confirmed on 2026** (upper bound +0.004, so
borderline; same direction as 2024 and 2025). **The constructor improvement seen in 2024-25 did not carry over**: constructors
are slightly worse in 2026. A post-hoc look (`phase5c`, not used for any decision) found the constructor loss is not
confined to the first weekends of the new regulations, so I do not know why.

## Decisions

* `ModelParams` defaults are unchanged: no simulator tuning was confirmed.
* The ensemble is **not** wired into `f1_season.py`: driver gain unconfirmed on 2026, constructor gain contradicted there.
* New simulator options (`team_block`, `block_link`, `p0_min/max`, `dnf_*`) stay in the code, off by default; a test checks that
  the defaults reproduce the previous model exactly.
* Harness numbers (continuous walk-forward) differ slightly from `f1_season.py`'s replay (which rebuilds the model for every
  weekend); compare within a table, not across.

## Not tried

Official `session_result` as the source of actual finishing order and DNF flags (cached now; would remove label noise from
post-race penalties); weather; per-circuit team strength; season-total (not weekend) objectives; the Final Fix chip.
