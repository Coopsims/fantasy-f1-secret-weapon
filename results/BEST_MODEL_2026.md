# Best 2026 model and season planner (overnight run, 2026-10-11/12)

## The model

Driver forecast = mean of two tuned tree models (extra-trees and XGBoost) on the full pre-lock feature panel, each
blended with the weekend simulator (its weight decays as 2026 data accumulate) and a little recent official form.
Constructor forecast = its two drivers' forecasts + its recent official bonus (qualifying bonus, pit stops, ...).
Spec: `results/s26_final_spec.json`; registered in MLflow as `f1-fantasy-points-2026` (alias in the registry).

What the tuning chose (2026 walk-forward, official points as truth): train on all history (no recency decay), give
2026 rows a little extra weight, train 2026 rows on official points (XGBoost) or reconstructed (extra-trees), simulator
weight about 0.3 early in the season falling to 0.06-0.17, recent form about 0.12-0.16.

## How good it is (honest numbers)

Every forecast uses only earlier weekends. Tuning used the odd 2026 gamedays; the numbers below are the even ones,
never used for tuning (8 weekends), against the game's official points (leak-free panel, see "Bugs found" below):

| | driver RMSE | MAE | rank corr | pick-5 value* |
|---|---|---|---|---|
| **This model (tuned extra-trees + XGBoost)** | **14.62** | **11.36** | 0.59 | **0.79** |
| Previous best (phase-7 trees + simulator) | 15.01 | 11.62 | 0.58 | 0.75 |
| Simulator + recent form (0.4) | 15.12 | 11.86 | 0.57 | 0.71 |
| Simulator alone | 15.22 | 12.02 | 0.56 | 0.71 |

\* actual points of the 5 drivers ranked highest, divided by the best possible 5.

Change vs the simulator: -0.60 RMSE (95% CI -1.26 to +0.05); vs simulator + form -0.50; tuned extra-trees alone vs the
previous best -0.36 (CI -0.72 to -0.04). Only 8 held-out weekends: the direction is consistent everywhere, the size is
uncertain.

Constructors (official points, both halves): derived from the drivers (sum + recent bonus) RMSE ~20.6 vs simulator
22.0-22.2 - the opposite of the 2024-25 finding, which was on reconstructed points.

## Planning (decisions) - a separate question

Season played walk-forward with each forecaster (official points and prices, 3-race rolling plan, exact solver):

| forecaster used for planning | season points | of the 5741 optimum |
|---|---|---|
| **0.5 x this model + 0.3 x simulator + 0.2 x recent form (planning blend)** | **4151** | **72%** |
| this model + 0.2 recent form | 4035 | 70% |
| simulator + recent form 0.4 | 3955 | 69% |
| this model alone | 3953 | 69% |
| simulator alone | 3330 | 58% |

Better forecasts did not by themselves win more points in this one season; shrinking them for planning did. Reason:
the optimiser chases the largest predicted edges, so noisy differences get over-acted on (the optimiser's curse);
shrinking toward the simulator and recent form damps that. The planners use the planning blend (it costs +0.16 RMSE as a
forecast, CI includes 0); `--sharp` plans on the raw model. Caveat: chosen among ~8 variants on the same season.

## Bugs found during the night
* **Leak (fixed):** the team track-affinity features included the teammate's result from the same weekend (two rows per
  team per weekend; "earlier in sort order" instead of "earlier weekend"). Found because the planner's future-race
  features did not match the panel. 2024-25 results with those features were ~0.2 RMSE too good; the 2026 numbers above
  are on the fixed panel (the extra-trees spec never used them). Earlier registered models are tagged superseded.
* Killing a worker-pool parent left its workers running (overloaded the machine); MLflow rejects "(" in metric names.

## What did not help
Tuning the simulator itself on 2026 (best odd setting -0.07, even +0.08); LightGBM (overfit: 15.10 held out);
random forest / CatBoost (14.81-14.91); chip reserve values in the rolling plan.

## The deployed version
Same structure re-tuned on all 17 gamedays (200 Optuna trials per family, warm-started, leak-free panel): XGBoost
(feature set without track features, trained on official points for 2026 rows, recency half-life 100 weekends) and
extra-trees (no track features, reconstructed points), averaged. In-sample RMSE 14.47 (not an honest number - use
14.62 above). Registered: `models:/f1-fantasy-points-2026@champion` (v3). Late in the season the simulator's weight
inside the forecast is ~0.01-0.02 (the trees carry it); the planning blend adds it back for decisions.

What drives it most (feature importance): qualifying form (team and driver qualifying-position EWMAs, qualifying Elo),
longer-run points form (slow EWMA, last-10 mean), race-position form, practice pace (for the next race, once practice
has run). The data-derived track features and similar-track affinities did not survive the leak fix as useful.

## Your next race (US GP, Austin; lock: Qualifying, Sat 24 Oct 21:00 UTC / 15:00 MDT)
From Hamilton, Antonelli, Colapinto, Lawson, Bearman, Ferrari, Racing Bulls (bank $2.4m, 3 free transfers):
OUT Hamilton, Colapinto, Racing Bulls - IN Mercedes, Hulkenberg, Lindblad, 2x on Antonelli: expected 214.5 vs 181.6
for keeping the team (no transfer penalty). Every model tried tonight gave this same move. Re-run after Friday's
practice (`python fetch_openf1_full.py` then `plan_next_race.py`) - practice pace is one of the stronger inputs.

Rest-of-season plans for each chip scenario: `results/SEASON_PLAN.md`.

