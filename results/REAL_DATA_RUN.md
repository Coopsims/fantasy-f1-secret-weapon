# First run on real OpenF1 data, updated after the data correction (2026-10-11)

Everything below was produced by the code in this branch on data fetched from OpenF1 that day: 2023-2026 (22 + 24 + 24 weekends of
history, 17 raced 2026 weekends; Sakhir and Jeddah 2026 were cancelled and Austin onwards had not been run). No synthetic data.
Raw API caches are gitignored; the CSVs in this folder are the program's outputs.

**This replaces an earlier version of this note.** The first version used data in which 2023-24 Sprint weekends had Qualifying and
Sprint Qualifying swapped and three 2023 weekends were missing; see `DATA_CORRECTION.md`. Numbers that changed are listed there.

## Data fixes needed before anything could be trusted

| Problem on real data | Effect | Fix |
|---|---|---|
| Driver list taken from Practice 1, which often runs reserve drivers | Race drivers silently dropped (Barcelona 2026: 15 of 22 cars) and some in the wrong team | Fetch each session's own line-up |
| Failed / rate-limited fetches cached as final (client returned `[]` for both "no data" and "failed") | 2024 Qatar and Abu Dhabi, 2025 Monza, 2026 Zandvoort and parts of 2023 missing | `get_strict` + per-endpoint status; incomplete weekends are retried |
| Weekends ordered by `meeting_key`, keyed by `(year, round)` | Relocated Kuala Lumpur race (2 Oct) sorted after Singapore and shared a round with Madrid | Store `date_start`, sort by date, unique rounds, calendar entry (round 18) |
| Replay horizon was `round + k` | Targeted cancelled Sakhir/Jeddah rounds | Horizon = next raced weekends |
| **Sprint weekends: Qualifying vs Sprint Qualifying told apart by a `session_key` order guess** | **Swapped in 2023-24 (12 weekends)** | **Use `session_name`** |

After the fixes all 17 raced 2026 weekends have Q and R (plus SQ and S on the 6 Sprint weekends), every race has a full grid and one
fastest lap, and overtakes average 2.6 (2024), 2.5 (2025) and 3.7 (2026) per driver per race.

Real facts in the data, not bugs: the 2026 DNF rate is 19.5% (about 4 retirements per race; stint-based and lap-based detection agree)
against about 10% in 2024-25; Hadjar is absent for rounds 14-16 (Lawson drove for Red Bull, Tsunoda for Racing Bulls); OpenF1 has
`stop_duration` for only about 7 of the 17 2026 races, so pit-stop points are known for those weekends only.

## 2025 walk-forward backtest (`f1_weekend_model.py backtest --years 2025`, 24 weekends)

| | Model | Last-3 baseline |
|---|---|---|
| Driver-total MAE | 9.35 | 10.77 |
| Rank correlation | 0.522 | 0.468 |

80% interval coverage 0.85. DNF Brier 0.0951 vs 0.0945 for a flat 7% guess (no better). Qualifying rank correlation 0.722.
Pole / winner top-pick hit rate 0.21 / 0.12 (24 weekends, so very noisy). A stronger naive baseline than last-3 exists:
an exponentially weighted average (alpha 0.2) scores 9.80 on the same weekends (see `MODEL_EXPERIMENTS.md`).

## Priors review (2024-25 evidence only; 2026 not used)

No prior was changed. Predicted DNF rate 12.2% vs actual 10.4%; the true fastest-lap holder gets on average 10.5% probability vs 5%
uniform; predicted overtake points 2.56 vs 2.51 per driver (these three checks were made on the pre-correction data; the
qualifying relabelling does not touch race DNF, fastest-lap or overtake labels). Per-track `overtaking` and `stops` priors correlate
positively with 2024-25 (Spearman 0.35 / 0.45); per-track DNF rate does not repeat from year to year (r about 0), so `chaos` cannot be
validated. Driver of the Day cannot be tested (OpenF1 does not publish it). `RELIABILITY_PRIOR` is a 2026 prior.

## 2026 replay (each weekend rebuilt only from earlier weekends)

Driver MAE 11.06 vs 12.44 (last-3), rank correlation 0.57, 80% coverage 0.84; constructor MAE 16.41 vs 18.13. Beats the baseline on
driver MAE in 14 of 17 weekends, on constructor MAE in 12 of 17. Weak spots: round 2 (rank corr 0.30) and round 8 (0.20).
Excludes Driver of the Day, and pit points where OpenF1 has no stop times.

| Round | Weekends of history | Driver MAE | Naive MAE | Rank corr | 80% cover | Constr MAE | Constr naive | Constr 80% cover |
|---|---|---|---|---|---|---|---|---|
| 1 | 70 | 11.47 | 14.45 | 0.58 | 0.77 | 22.07 | 24.93 | 0.45 |
| 2 | 71 | 16.09 | 19.36 | 0.30 | 0.59 | 29.41 | 22.09 | 0.27 |
| 3 | 72 | 8.61 | 11.23 | 0.60 | 0.82 | 16.80 | 17.95 | 0.64 |
| 6 | 73 | 12.82 | 15.30 | 0.41 | 0.73 | 20.88 | 26.09 | 0.82 |
| 7 | 74 | 12.39 | 11.68 | 0.46 | 0.82 | 10.04 | 14.15 | 1.00 |
| 8 | 75 | 16.43 | 16.77 | 0.20 | 0.59 | 11.21 | 21.12 | 0.91 |
| 9 | 76 | 11.47 | 12.55 | 0.66 | 0.91 | 12.60 | 14.88 | 0.91 |
| 10 | 77 | 8.51 | 9.74 | 0.78 | 1.00 | 13.68 | 13.73 | 0.82 |
| 11 | 78 | 9.67 | 13.08 | 0.66 | 0.91 | 14.09 | 20.06 | 1.00 |
| 12 | 79 | 8.10 | 9.59 | 0.64 | 0.95 | 11.76 | 10.73 | 1.00 |
| 13 | 80 | 7.99 | 8.56 | 0.65 | 0.91 | 12.49 | 12.15 | 0.82 |
| 14 | 81 | 9.44 | 10.38 | 0.68 | 0.82 | 15.30 | 14.55 | 0.82 |
| 15 | 82 | 10.25 | 12.15 | 0.66 | 0.91 | 19.34 | 21.79 | 0.91 |
| 16 | 83 | 8.21 | 8.12 | 0.68 | 0.95 | 11.13 | 13.36 | 1.00 |
| 17 | 84 | 12.87 | 14.95 | 0.51 | 0.91 | 22.45 | 22.67 | 0.73 |
| 18 | 85 | 10.95 | 12.86 | 0.57 | 0.86 | 18.20 | 22.06 | 0.73 |
| 19 | 86 | 12.76 | 10.68 | 0.60 | 0.86 | 17.46 | 15.85 | 0.82 |

## Season (17 weekends, `f1_season.py all --year 2026`)

| | Points |
|---|---|
| Hindsight optimum (proven optimal, HiGHS status 7) | 5083.0 |
| Model, no future knowledge | 2753.0 (54% of optimum) |
| Your recorded team, rounds 1-2 only (history file has 2 rows) | 512.0 |

Like-for-like on rounds 1-2: optimum 651, model 391, you 512.

**Do not read the model total as a stable measure.** Before the data correction the same procedure gave 3218 (63%) although forecast
accuracy was almost identical (driver MAE 11.12 vs 11.06): small forecast differences changed which team the rolling optimiser picked
and which chips it used, moving the season total by 465 points. It is one season of 17 weekends. All totals are on estimated prices
for 14 of 17 weekends (`fantasy_f1_prices.csv` stops at round 2), a 100m starting budget, reconstructed points (no Driver of the Day, pit
points only where known) and the transfer rules below.

## Not confirmed

Transfer rules (2 free per weekend, bank up to 3, -10 per extra, wildcard / 3x booster / no-negative once each) are unchanged and
**not checked against the real game**. Final Fix, Autopilot and any Limitless-style chip are not modelled. The game's round numbering
after the cancellations and the Kuala Lumpur race is unknown, which matters when price / history CSVs are extended past round 2.
