# Can the forecast be improved? (experiments of 2026-10-11, all logged in MLflow)

Everything here was run with `mlflow_experiments.py` and the scripts in `experiments/`; 271 runs are in the
MLflow experiment `f1-weekend-forecasting` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`). The store itself is gitignored.

## Protocol

* Walk-forward: a forecast for weekend *t* sees only weekends before *t* (tests scramble the target weekend and everything
  after it and check the predictions do not move).
* **dev = 2024** (tuning), **val = 2025** (confirmation, never tuned on), **test = 2026** (held out; scored once, after a
  pre-registration run fixed the candidate and the decision rule).
* Differences are paired bootstraps over weekends (5000 resamples); "CI" below is the 95% interval of the MAE change.
  Weekends are treated as independent, and each split has only 24 (test: 17) of them, so small effects are hard to see.
* Data: 2023 (partial: OpenF1 has no results for Imola, Spielberg and São Paulo), 2024, 2025, 2026. Metric = driver fantasy
  points per weekend (no Driver of the Day), as in the earlier backtest.

## Results

| Forecaster | dev 2024 MAE | val 2025 MAE | note |
|---|---|---|---|
| last-1 / last-3 / last-5 mean | 10.64 / 8.91 / 8.77 | 13.26 / 10.78 / 10.19 | |
| season-to-date mean | 8.46 | 10.13 | |
| EWMA (alpha 0.2) | 8.25 | 9.81 | strongest naive baseline |
| Simulator, defaults | 8.05 | 9.35 | |
| Gradient-boosted regression (depth 2, 100 trees, lr 0.03) | 7.94 | 9.18 | settings picked on 2024 |
| **Simulator + GBM, adaptive weights (NNLS on earlier weekends)** | 7.85 | 9.13 | change vs simulator -0.21 / -0.21, CIs exclude 0 |
| Simulator 40% + GBM 60% (weight picked on 2024) | 7.84 | 9.11 | -0.22 / -0.23, CIs exclude 0 |

What did **not** help (details in MLflow, phases 3a, 3b, 4):

* Time-varying team strength (`team_block`): flat or slightly better on 2024 (best 8.03 vs 8.05), worse on 2025 (9.35 to 9.55).
* DNF settings (`p0_max`, shrinkage, half-life): all within about +-0.02 MAE.
* 96 random draws over 12 simulator settings: best on 2024 was 7.97 (vs 8.05); the five best re-run on 2025 changed MAE
  by -0.03 to +0.05, every CI including 0. The search pushed `half_life` to the bottom of its range (about 4.5) without any
  confirmed benefit.
* Adding EWMA to the ensemble; feeding the simulator's forecast into the GBM as features (no better than the GBM alone).

## Held-out 2026 (17 weekends, scored once)

| | driver MAE | change vs simulator (95% CI) | constructor MAE | change vs simulator (95% CI) |
|---|---|---|---|---|
| Simulator, defaults | 11.00 | | 15.92 | |
| Simulator + GBM, adaptive (pre-registered primary) | 10.82 | -0.18 [-0.37, +0.02] | 16.31 | +0.39 [-0.01, +0.81] |
| 40/60 blend | 10.80 | -0.20 [-0.45, +0.07] | 16.47 | +0.56 [+0.01, +1.15] |
| GBM alone | 10.89 | -0.11 [-0.50, +0.31] | | |
| EWMA 0.2 / last-3 | 11.21 / 12.21 | +0.21 / +1.21 | | |

By the rule fixed beforehand (CI must exclude 0) **the driver improvement is not confirmed on 2026**: the direction matches
2024 and 2025 and the ensemble is better in 13 of 17 weekends (3 worse, 1 tie), but the interval includes 0. **The constructor improvement seen
in 2024-25 (about -0.6 MAE, CIs excluding 0) did not carry over; constructors were slightly worse.** A post-hoc look
(`phase5c`, not used for any decision) found no sign that the loss is limited to the first weekends of the new regulations,
so I do not know why.

## Decisions

* `ModelParams` defaults are unchanged: nothing in the simulator tuning was confirmed.
* The ensemble is **not** wired into `f1_season.py`: driver gain unconfirmed on 2026, constructor gain contradicted.
* New simulator options (`team_block`, `block_link`, `p0_min/max`, `dnf_*`) stay in the code, off by default, with a test that the
  defaults reproduce the previous model exactly.
* 2026 numbers here (simulator 11.00 driver MAE) differ slightly from the season replay (11.12) because the harness
  recalibrates on a fixed cadence; compare within a table, not across.

## Not tried

Practice-session pace as a feature (OpenF1 has the data, but whether it is available before the team lock depends on the
game's deadline, which I have not checked); weather; per-circuit team strength; season-total (not weekend) objectives.
