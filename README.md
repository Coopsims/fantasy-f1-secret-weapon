# Fantasy F1 Secret Weapon

Predicts Fantasy F1 points for **every scoring event of a race weekend** –
qualifying, sprint qualifying → sprint, and the race – for every driver and
constructor, as full probability distributions, and feeds them into the
transfer / chip optimizer.

## Quick start

```bash
pip install -r requirements.txt

python f1_weekend_model.py demo                       # end-to-end on synthetic data, no network
python f1_weekend_model.py forecast --round 19        # every event for a weekend
python f1_weekend_model.py forecast --round 19 --driver Norris   # + component breakdown
python f1_weekend_model.py forecast --round 19 \
       --quali "Norris,Piastri,Russell,..."            # condition the race on the real grid
python f1_weekend_model.py forecast --round 19 --penalty Hamilton=3 --exclude Stroll
python f1_weekend_model.py backtest --years 2025       # walk-forward accuracy on real history
python f1_weekend_model.py tune --years 2025           # tune half-life / prior trust
python f1_season.py all --year 2026                     # replay + play + hindsight-optimal season
python f1_season.py all --demo                          # same, on synthetic data
python f1_season.py tune --year 2026 --workers 8        # search play policies for the smallest gap to the optimum
python fantasy_optimizer.py                            # interactive optimizer (menu 6 = weekend forecast)
python -m pytest tests                                 # unit + integration tests
```

Data comes from the files the trainer already writes (`raw_historical_data.json`,
`raw_2026_results.json`; falls back to `f1_driver_training_data.csv`).
With no data at all the model starts from the pre-season price prior.

## What gets predicted

For each driver: **Qualifying, Sprint, Race** points, with components
(finish, positions gained/lost, overtakes, fastest lap, DNF, Driver of the Day),
mean / sd / 10-50-90 % range, and probabilities of pole, win, podium, points,
DNF and fastest lap.
For each constructor: the same per event, plus the Q2/Q3 progression bonus and
pit-stop points (P(both cars in Q3) etc.).
Team-level: the distribution of *your* team's score, the expected value of each
boost choice (2x / 3x) and what the **No Negative** chip is worth this weekend.

## How the model works (`f1_weekend_model.py`)

1. **Ratings.** Every past session becomes normal-score outcomes, fitted with a
   time-decayed ridge regression: `team-season effect + driver effect (+ grid
   effect in races)`. Qualifying pace and race pace are rated separately, and race
   pace is measured *net of grid position*. Team effects of earlier seasons are
   nuisance terms, so only **driver skill** carries over the 2026 regulation reset;
   2026 teams start from a prior (pre-season constructor prices) and are re-learned
   as real rounds arrive (half-life 10 weekends by default).
2. **Calibration.** A walk-forward pass (fit on the past, predict the next weekend)
   measures how noisy sessions really are and how correlated team-mates are, so
   simulated spreads are calibrated rather than guessed.
3. **Simulation** (10k draws): qualifying order → sprint grid → sprint →
   race grid (with optional penalties) → DNFs (team reliability × driver
   multiplier × a weekend "chaos" factor) → finishing order (grid stickiness set
   per circuit: Monaco ≫ Baku) → overtakes → fastest lap → pit stops → Driver of
   the Day. Team-mates share a car/weekend shock, which is what makes constructor
   variance realistic.
4. **Scoring** uses the exact rules in `f1_rules.py` on every draw.
5. **Conditioning.** Pass real results (`known=`; `--quali`; or let the optimizer
   auto-detect finished sessions on OpenF1) and the rest of the weekend is
   re-forecast on the true grid – e.g. before spending a Final Fix.

### Insights it surfaces
* Starting on pole can never gain places, so a slower car there bleeds points
  (−1 per place lost) even as its win probability rises – the simulator prices that in.
* Chaotic circuits raise both DNF risk and variance, so they favour boosting a
  high-upside driver and make the No Negative chip worth more.

## Files

| file | role |
|---|---|
| `f1_rules.py` | scoring constants + vectorised scoring (single source of truth) |
| `f1_calendar.py` | grid, calendar, name canonicalisation (Hülkenberg, Sauber→Audi), circuit profiles |
| `f1_weekend_data.py` | raw OpenF1 / CSV → tidy session table; on-track overtakes from the position stream; scores real weekends |
| `f1_weekend_model.py` | ratings, calibration, simulator, forecasts, backtest, CLI |
| `f1_synthetic.py` | synthetic seasons with known truth (tests, demo) |
| `f1_ml_predictor.py` | `predict_*` API used by the optimizer; back-ends `sim` (default) / `xgb` / `blend` via `F1_PREDICTOR` |
| `f1_trainer.py` | original XGBoost pipeline (still works) |
| `f1_prices.py` | rebuilds weekend-by-weekend prices with the game's price rule (3-race points-per-$m average, tiers at $19m, cut-offs 0.6 / 0.9 / 1.2), fitted to any real prices you have |
| `f1_strategy.py` | exact season optimiser (MILP, validated against exhaustive enumeration): teams, boosts, transfers, free-transfer banking, wildcard / 3x / no-negative |
| `f1_season.py` | walk-forward replay of a season (rebuilt each weekend from earlier weekends only), live-play simulation, hindsight optimum |
| `openf1_utils.py` | OpenF1 client (token from `.env`: `OPENF1_USERNAME` / `OPENF1_PASSWORD`) |
| `fantasy_optimizer.py` | transfers, chips, price changes; now volatility-, chip- and risk-aware |

## Prices and the play policy

Prices for weekends you did not record are rebuilt by `f1_prices.py` from the published
2025 rule (FanAmp guide): tiers split at $19m (A: ±0.1/±0.3, B: ±0.2/±0.6); rating from the
average points per $m over the last three races (each race's points over the price paid for
it), cut-offs 0.6 / 0.9 / 1.2, with the early-season zero padding. Every real price in
`fantasy_f1_prices.csv` is used as a checkpoint: the rule's cut-offs and tier split are fitted
to reproduce them and the reconstruction snaps back to them. The run prints how well the fitted
rule reproduces your real prices. Add the game's *current* prices as the latest round (optimizer
menu 4 → scrape) to pin the end of the season too.

`f1_season.py tune` replays the season once (cached in `season_<year>_replay.pkl`) and then plays it
under many decision policies – horizon, discount, projected prices, value of team-value growth,
chip reserve, forecast shrinkage – reporting each one's gap to the proven optimum. Picking the
best policy on the same 17 weekends flatters it; the output also shows the gap when the policy is
chosen on the first half of the season and scored on the second.

## Validation – read this

There was no access to OpenF1 while this was built, so **it has not been
backtested on real 2024/2025 data**. What has been verified:

* 38 unit/integration tests: scoring rules, data conversion, simulator
  invariants (permutations, one winner/fastest lap/DotD per race, constructor =
  sum of drivers), conditioning, penalties, chip maths.
* On synthetic seasons with a known truth (3 seeds, last 12 weekends of 2026) the
  fitted ratings recover it (corr > 0.85), the calibrated noise scale lands near
  the true value, and a walk-forward backtest beats a "last 3 results" baseline:
  driver-total MAE ≈ 8.0 vs 9.3, rank-correlation ≈ 0.54 vs 0.46, qualifying
  rank-correlation ≈ 0.79, and the 80 % intervals cover ≈ 86 % (slightly wide –
  the simulator deliberately adds circuit "chaos" variance the synthetic world lacks).

That shows the machinery is sound – not that F1 is as predictable as the
synthetic world. **Run `python f1_weekend_model.py backtest --years 2025`
on your data before trusting it**, then `tune`. Things to check/edit there:
circuit profiles (`TRACK_PROFILES`), team reliability and pit-time priors
(`RELIABILITY_PRIOR`, `PIT_PRIOR`), the Driver-of-the-Day heuristic and the
fastest-lap preference (`ModelParams.fl_beta`).

## Bugs found and fixed in the existing code

* `add_round_numbers` ordered rounds **alphabetically by circuit name**, scrambling
  every lag/rolling feature (and "previous race") – now chronological.
* Overtakes were **always 0**: the code read `position` from OpenF1's `laps`
  endpoint, which has none. Now derived from the position stream with pit-cycle
  shuffles excluded.
* Q2 cut-off was 15; with 22 cars **16** progress (constructor bonus tiers).
* Accented names (`Hülkenberg`, `Pérez`) and team renames (`Kick Sauber`→Audi,
  `RB`→Racing Bulls) didn't match the 2026 grid, so those drivers/teams lost
  their history.
* `fetch_2026_results` cached a race weekend as soon as it had a (scheduled)
  Race session – before the race – freezing it as empty. Incomplete weekends
  are no longer cached.
* Pit points: prefers `stop_duration` (stationary) over `pit_duration` (whole
  lane, ~20 s, which can never score).

Re-run `python f1_trainer.py` to regenerate training data with these fixes.

## Experiments and MLflow

`mlflow_experiments.py` is a walk-forward evaluation harness (simulator, naive baselines, gradient-boosted model, blends)
that logs every run to MLflow; `experiments/` holds the scripts that produced `results/MODEL_EXPERIMENTS.md`.

```bash
python experiments/phase1_baselines.py            # reference forecasters on 2024 (dev) and 2025 (val)
mlflow ui --backend-store-uri sqlite:///mlflow.db  # browse params, metrics, per-weekend curves, predictions
```

## Data

```bash
python fetch_openf1_full.py          # one sequential pull of everything useful per weekend -> openf1_full/ (~25 min, 39 MB)
python rebuild_raw_from_full.py      # regenerate raw_historical_data.json / raw_2026_results.json from it (no network)
```

Read `docs/OPENF1_NOTES.md` first: rate limits (pace requests at about 1 s, never in parallel), the failed-request-looks-empty
trap, and the Sprint-weekend session labelling. `results/DATA_CORRECTION.md` explains a labelling bug that affected 2023-24.
