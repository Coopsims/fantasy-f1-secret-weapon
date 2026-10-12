# Best 2026 model and season planner (overnight run, 2026-10-11/12)

## The model

Driver forecast = mean of two tuned tree models (extra-trees and XGBoost, 5 random seeds each) on the full pre-lock
feature panel, each blended with the weekend simulator (its weight decays as 2026 data accumulate) and a little recent
official form. Constructor forecast = its two drivers' forecasts + its recent official bonus (qualifying bonus, pit
stops, ...). Spec: `results/s26_final_spec.json`; registered in MLflow as `models:/f1-fantasy-points-2026@champion` (v4).

What the tuning chose (2026 walk-forward, official points as truth): train on all history, give 2026 rows a little
extra weight, simulator weight 0.6-1.0 at the start of the season falling to ~0.02-0.03 by now, recent form small.

## How good it is (honest numbers)

Every forecast uses only earlier weekends. Hyperparameters were tuned on the odd 2026 gamedays; the table is the even
ones (8 weekends) against the game's official points, leak-free panel:

| | driver RMSE | MAE | rank corr | pick-5 value* |
|---|---|---|---|---|
| **Deployed: extra-trees + XGBoost** | **14.62** (honest: 14.65-14.75, see below) | 11.36 | 0.59 | 0.79 |
| extra-trees alone | 14.65 | 11.41 | 0.60 | 0.78 |
| kNN / elastic-net / MLP (tuned the same way tonight) | 14.92 / 14.97 / 15.06 | | | |
| Previous best (phase-7 trees + simulator) | 15.01 | 11.62 | 0.58 | 0.75 |
| Simulator + recent form (0.4) | 15.12 | 11.86 | 0.57 | 0.71 |
| Simulator alone | 15.22 | 12.02 | 0.56 | 0.71 |

\* actual points of the 5 drivers ranked highest, divided by the best possible 5.

**Selection caveat (corrected tonight).** The even gamedays were not used for tuning, but the *pair* extra-trees +
XGBoost was picked after looking at them - it is the best of 92 family subsets there, so 14.62 is the most flattering
number. Choosing the subset on the odd gamedays instead (`experiments/s26_selection_check.py`) gives 14.65 over the
original 5 families (it picks extra-trees alone) and 14.75 over all 8 (it picks extra-trees + MLP, because the MLP
fitted its tuning half: 14.58 odd -> 15.06 even). I did not deploy that pick; ET+XGB stays. Honest expectation:
**~14.65-14.75, about 0.5 better than the simulator (15.22)** (extra-trees vs simulator -0.57, 95% CI -1.22 to +0.10;
8 weekends - the direction is consistent, the size uncertain).

Constructors (official points, both halves): derived from the drivers (sum + recent bonus) RMSE ~20.6 vs simulator
22.0-22.2.

Seed averaging (5 seeds per family): RMSE unchanged (-0.001, CI -0.027 to +0.022); a single seed moves a driver's
forecast by ~0.3 points and two seeds share 4.82 of their top 5 - deployed to keep seed luck out of close decisions.

## Planning (decisions) - a separate question

Season played walk-forward with each forecaster (official points and prices, 3-race rolling plan, exact solver),
re-scored tonight under the corrected game rules (below):

| forecaster used for planning | season points | of the proven 5778 optimum |
|---|---|---|
| **0.5 x this model + 0.3 x simulator + 0.2 x recent form (planning blend)** | **4143** | **72%** |
| simulator + recent form 0.4 | 4081 | 71% |
| this model alone | 4035 | 70% |
| simulator alone | 3456 | 60% |

(Before the rule fixes these were 4151 / 3955 / 3953 / 3330 of 5741: the x3 fix raised every policy that plays the chip.)

Shrinking the forecasts for planning wins: the optimiser chases the largest predicted edges, so noisy differences get
over-acted on (the optimiser's curse). The planning blend is best in every policy setting tried (4095-4143 points
across discount 0.6-1.0 and chip reserve 0-30, the raw model 3750-4048), but its lead over simulator + form is within
the season-to-season noise. The planners use the planning blend; `--sharp` plans on the raw model. In-sample, one
season.

## Game rules (checked against the official FAQ tonight; two fixes)
* 2 free transfers per race, one unused carries over (max 3), -10 per extra; transfers counted as the net change; one
  chip per race - all as modelled.
* **Fix 1 - x3 Boost:** the chip puts x3 on one driver while the regular x2 stays on another driver. The solver had
  x3 on the boosted driver only, undervaluing the chip by about one driver's score. Fixed everywhere (solver, scoring,
  brute-force check, replays, planners), with tests.
* **Fix 2 - Limitless:** no free transfers carry into the race after a Limitless week. Fixed, with a test.
* Final Fix: one change between Qualifying and the Grand Prix -> new `final_fix.py` (below). Table:
  `docs/FANTASY_FEED_NOTES.md`.

## Bugs found during the night
* **Leak (fixed):** the team track-affinity features included the teammate's result from the same weekend. 2024-25
  results with those features were ~0.2 RMSE too good; the 2026 numbers above are on the fixed panel. Earlier
  registered models are tagged superseded.
* The two rule fixes above; the ensemble-selection caveat above.
* Killing a worker-pool parent left its workers running; MLflow rejects "(" in metric names.

## What did not help
Tuning the simulator on 2026 (best odd -0.07, even +0.08); LightGBM (15.10 held out); random forest / CatBoost
(14.81-14.91); kNN, elastic-net, MLP (above); stacking all 8 families with weights fitted on the odd gamedays (even
14.70 - it leans on the over-fitted MLP); a calibration scale (x1.075 fitted on odd: even RMSE -0.06, CI -0.21 to
+0.10, but the season replay got worse, 4143 -> 4114); chip reserve values in the rolling plan; conditioning the race
forecast on the real grid (RMSE 13.18 -> 13.15, CI -0.21 to +0.13 - see Final Fix).

What drives the model (feature importance): qualifying form (team and driver qualifying-position EWMAs, qualifying
Elo), longer-run points form, race-position form, practice pace (for the next race, once practice has run). The track
features did not survive the leak fix as useful.

## Your next race (US GP, Austin; lock: Qualifying, Sat 24 Oct 21:00 UTC / 15:00 MDT)
From Hamilton, Antonelli, Colapinto, Lawson, Bearman, Ferrari, Racing Bulls (bank $2.4m, 3 free transfers):
**OUT Hamilton, Colapinto, Racing Bulls - IN Mercedes, Hulkenberg, Lindblad, 2x on Antonelli: expected 212.4 vs 181.4
for keeping the team** (no penalty). Replaying the model's own errors from each of the 17 weekends so far, the move
beats keeping the team in 82% of them; this week's range 184-344 (10th-90th percentile). Every model tried tonight
gave this same move. Re-run after Friday's practice - practice pace is one of the stronger inputs.

Rest-of-season plans for each chip scenario (`results/SEASON_PLAN.md`; which chips you have left is unknown to me):

| chips available | expected points, 6 races | where the plan plays them |
|---|---|---|
| none | 994 | - |
| 3x Boost | 1036 | Sao Paulo |
| No Negative | 1006 | US GP |
| Wildcard | 996 | Las Vegas |
| Limitless | 1050 | Mexico City |
| all four | 1105 | No Negative US GP, Limitless Mexico City, Wildcard Sao Paulo, 3x Las Vegas |

Autopilot is worth about 6 points in a typical week (play it in a week with no other chip). Chip timing is a close
call: `plan_next_race.py` (3 races ahead, later races weighted 0.6) can prefer playing a chip now where the full-season
plan waits; the 2026 replay could not separate those settings (4131-4143).

## How to use it
Any team, any prices (default: the game's current prices from its feed; override any with `--price NAME=V`):

    python fetch_fantasy_feed.py          # after each race: official points + prices
    python fetch_openf1_full.py           # after each session (practice adds to the forecast; lock is Qualifying)

    # best moves for the next race from your team (exact solver, plans 3 races ahead, acts on the first)
    python plan_next_race.py --team Hamilton Antonelli Colapinto Lawson Bearman Ferrari "Racing Bulls" \
        --bank 2.4 --free 3 --chips 3x_booster limitless autopilot [--keep X] [--avoid Y] [--price Antonelli=26.5]

    # the best line for the rest of the season from any team (or --budget 100 for a fresh team)
    python plan_season.py --team ... --bank 2.4 --free 3 --chips wildcard limitless [--csv plan.csv]

    # after Qualifying, if you hold Final Fix: the best single swap for the Grand Prix
    python final_fix.py --team ... --boost Antonelli --bank 0.3 [--not-starting X] [--penalty Y=5]

The first run after new data refits the 10 models (~1 minute); later runs use the cache (a few seconds).
