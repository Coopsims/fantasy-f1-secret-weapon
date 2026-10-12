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
never used for tuning (8 weekends), against the game's official points:

| | driver RMSE | MAE | rank corr | pick-5 value* |
|---|---|---|---|---|
| **This model** | **14.59** | **11.32** | 0.59 | **0.81** |
| Previous best (phase-7 trees + simulator) | 15.00 | 11.60 | 0.58 | 0.73 |
| Simulator + recent form (0.4) | 15.12 | 11.86 | | 0.71 |
| Simulator alone | 15.22 | 12.02 | 0.56 | 0.71 |

\* actual points of the 5 drivers the model ranks highest, divided by the best possible 5.

Change vs the simulator: -0.63 RMSE (95% CI -1.29 to +0.01); vs simulator + form -0.52 (-1.10 to +0.14); the single
tuned extra-trees beat the previous best with a CI excluding 0 (-0.33, -0.68 to -0.02). 8 weekends is few: the
direction is consistent across every comparison, the size is uncertain.

Constructors (official points, both halves): derived from the drivers RMSE ~20.6 vs simulator 22.0-22.2.

Season played walk-forward with each forecaster (official points and prices, 3-race rolling plan): this model 3971
(69% of the proven hindsight optimum 5741), simulator + form 3955, simulator 3330. Decision settings (discount, chip
reserve) move totals by +-50, i.e. noise for one season.

## What did not help
Tuning the simulator itself on 2026 (best odd setting -0.07, even +0.08); LightGBM (overfit: 15.10 held out);
random forest / CatBoost (14.81-14.91); chip reserve values in the rolling plan.

(continued below with the deployment tuning and the next-race plan)
