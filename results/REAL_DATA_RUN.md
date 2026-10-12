# First run on real OpenF1 data (run on 2026-10-11)

Everything below was produced by the code in this branch on data fetched from OpenF1 that day:
2024-25 (48 race weekends) and 2026 (17 raced weekends; Sakhir and Jeddah were cancelled, and
Austin onwards had not been run). No synthetic data was used. Raw API caches are gitignored;
the CSVs in this folder are the program's outputs.

## Data fixes needed before anything could be trusted

| Problem on real data | Effect | Fix |
|---|---|---|
| Driver list was taken from Practice 1, which often runs reserve drivers | Race drivers silently dropped (e.g. Barcelona 2026: 15 of 22 cars; Mexico City 2025: 11 of 20) and some drivers put in the wrong team | Fetch each session's own line-up (`get_session_data`), backfill old caches |
| Failed / rate-limited fetches were cached as final | 2024 Qatar, 2024 Abu Dhabi, 2025 Monza, 2026 Zandvoort missing; several races without laps/stints/pits | `meeting_gaps()`; incomplete meetings are retried, not trusted |
| Weekends ordered by `meeting_key` and keyed by `(year, round)` | The relocated Kuala Lumpur race (2 Oct, key 1308) sorted after Singapore and shared "round 16" with Madrid, merging two weekends | Store `date_start`, sort by date, force unique rounds, add the race to the calendar (round 18; later rounds shift by one) |
| Replay horizon was `round + k` | Targeted the cancelled Sakhir/Jeddah rounds as if they were the next weekend | Horizon = next raced weekends |

After the fixes: all 17 raced 2026 weekends have Q and R (plus SQ and S on the 6 sprint weekends), every race has
a full grid and exactly one fastest lap, overtakes average 2.6 (2024), 2.5 (2025) and 3.7 (2026) per driver per race.

Facts in the data that are real, not bugs: 2026 DNF rate is 19.5% (about 4.1-4.3 retirements per race; stint-based and
lap-based detection agree) against about 10% in 2024-25; Hadjar is absent for rounds 14-16 (Lawson drove for Red
Bull, Tsunoda for Racing Bulls). OpenF1 provides `stop_duration` for only about 7 of 17 2026 races, so pit-stop
points are known for those weekends only.

## 2025 walk-forward backtest (`f1_weekend_model.py backtest --years 2025`, 24 weekends)

| | Model | Last-3 baseline |
|---|---|---|
| Driver-total MAE | 9.36 | 10.78 |
| Rank correlation | 0.517 | 0.467 |

80% interval coverage 0.84. DNF Brier 0.0955 vs 0.0945 for a flat 7% guess (no better). Qualifying rank correlation 0.719. Pole / winner top-pick hit rate 0.25 / 0.17.
`tune --years 2025` (nine settings of `half_life` x `team_ridge_new`): MAE spread 9.51-9.57, i.e. noise; defaults kept.

## Priors review (2024-25 evidence only; 2026 not used)

No prior was changed. Evidence: predicted DNF rate 12.2% vs actual 10.4%; the true fastest-lap holder gets on average
10.5% probability vs 5% uniform; predicted overtake points 2.56 vs 2.51 actual per driver. Per-track `overtaking` and
`stops` priors correlate positively with 2024-25 (Spearman 0.35 / 0.45); per-track DNF rate does not repeat from
year to year (r about 0), so `chaos` cannot be validated or improved from this data. Driver of the Day cannot be
tested (OpenF1 does not publish it). `RELIABILITY_PRIOR` is a 2026 prior and cannot be tested on 2025.

## 2026 replay (each weekend rebuilt only from earlier weekends)

Driver MAE 11.12 vs 12.44 (last-3), rank correlation 0.57, 80% coverage 0.84; constructor MAE 16.46 vs 18.13.
Model beats the baseline on driver MAE in 14 of 17 weekends, on constructor MAE in 12 of 17.
Weak spots: round 2 (rank corr 0.30, coverage 0.59) and round 8 (0.20, 0.55).
Accuracy excludes Driver of the Day, and pit points where OpenF1 has no stop times.

| Round | Weekends of history | Driver MAE | Naive MAE | Rank corr | 80% cover | Constr MAE | Constr naive | Constr 80% cover |
|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 11.49 | 14.45 | 0.56 | 0.77 | 22.22 | 24.93 | 0.45 |
| 2 | 49 | 16.18 | 19.36 | 0.30 | 0.59 | 29.71 | 22.09 | 0.36 |
| 3 | 50 | 8.65 | 11.23 | 0.61 | 0.82 | 16.94 | 17.95 | 0.73 |
| 6 | 51 | 12.84 | 15.30 | 0.39 | 0.73 | 20.87 | 26.09 | 0.82 |
| 7 | 52 | 12.58 | 11.68 | 0.48 | 0.82 | 10.36 | 14.15 | 1.00 |
| 8 | 53 | 16.45 | 16.77 | 0.20 | 0.55 | 11.12 | 21.12 | 0.91 |
| 9 | 54 | 11.54 | 12.55 | 0.66 | 0.91 | 12.50 | 14.88 | 0.91 |
| 10 | 55 | 8.49 | 9.74 | 0.79 | 1.00 | 13.37 | 13.73 | 0.82 |
| 11 | 56 | 9.94 | 13.08 | 0.68 | 0.91 | 14.50 | 20.06 | 1.00 |
| 12 | 57 | 8.17 | 9.59 | 0.64 | 0.95 | 11.50 | 10.73 | 1.00 |
| 13 | 58 | 8.03 | 8.56 | 0.65 | 0.91 | 12.46 | 12.15 | 0.82 |
| 14 | 59 | 9.48 | 10.38 | 0.67 | 0.82 | 15.47 | 14.55 | 0.82 |
| 15 | 60 | 10.35 | 12.15 | 0.66 | 0.91 | 19.66 | 21.79 | 0.91 |
| 16 | 61 | 8.26 | 8.12 | 0.67 | 0.95 | 10.96 | 13.36 | 1.00 |
| 17 | 62 | 12.91 | 14.95 | 0.51 | 0.91 | 22.58 | 22.67 | 0.64 |
| 18 | 63 | 11.07 | 12.86 | 0.57 | 0.86 | 18.38 | 22.06 | 0.73 |
| 19 | 64 | 12.66 | 10.68 | 0.63 | 0.86 | 17.24 | 15.85 | 0.82 |

## Season (17 weekends, `f1_season.py all --year 2026`)

| | Points |
|---|---|
| Hindsight optimum (proven optimal, HiGHS status 7) | 5083.0 |
| Model, no future knowledge | 3218.0 (63% of optimum) |
| Your recorded team, rounds 1-2 only (history file has 2 rows) | 512.0 |

Like-for-like on rounds 1-2: optimum 651, model 155, you 512.

**These totals are on estimated prices for 14 of 17 weekends** (`fantasy_f1_prices.csv` stops at round 2, history at
round 2), a 100m starting budget, reconstructed points (no Driver of the Day, pit points only where known), and
the transfer rules below. Treat them as indicative, not as game scores.

## Not confirmed

Transfer rules (2 free per weekend, bank up to 3, -10 per extra, wildcard / 3x booster / no-negative once each) are
unchanged and **not checked against the real game**. Final Fix, Autopilot and any Limitless-style chip are not modelled.
The game's own round numbering after the cancellations and the Kuala Lumpur race is unknown, which matters when
price / history CSVs are extended past round 2.
