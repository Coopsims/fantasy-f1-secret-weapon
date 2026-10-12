# Official F1 Fantasy feed (fantasy.formula1.com) - what it gives and what was checked

`fetch_fantasy_feed.py` reads the game's public JSON feeds (paced at 1.5 s, each completed gameday fetched once):

* `feeds/schedule/raceday_en.json`: gamedays. The game numbers **raced** weekends 1, 2, 3 ... (cancelled races skipped:
  Miami 2026 = gameday 4, the relocated Kuala Lumpur race = "Bahrain Grand Prix", gameday 16). Each gameday's first
  session start is the team lock (Qualifying; on Sprint weekends the Sprint, which the game labels "Sprint Qualifying").
* `feeds/drivers/{gameday}_en.json`: every asset with `Value` (price for that gameday), `GamedayPoints` (official
  points, incl. Driver of the Day and pit stops) and `SessionWisePoints`. The game creates a separate asset per
  (driver, team): 2026 has "Liam Lawson (Red Bull Racing)" for his stand-in rounds and "Yuki Tsunoda (Racing Bulls)";
  the model's driver is the asset of his 2026 team, the others are skipped.
* `feeds/popup/playerstats_<id>.json` answered 403 for the id tried; not used.

## Checks (2026, after Singapore)

* Screenshot of the user's team: all 7 prices equal the feed's gameday-18 (Austin) prices.
* The user's `fantasy_f1_prices.csv`: rounds 1 and 2 equal the official prices (33/33); round 0 (pre-season, the
  hard-coded defaults) differs for 4 of 33.
* Official vs this project's reconstructed points: correlation 0.97; MAE 3.1 per driver and 6.7 per constructor per
  weekend (ours have no Driver of the Day and pit points only where OpenF1 has stop times).
* `AdditionalStats` (dotd_pts, fastest_lap_pts, ...) are season-to-date totals whose snapshots are inconsistent between
  gameday files (differencing them gives e.g. 90 Driver-of-the-Day points one weekend and none for eight) - not used.

## Price rule against the official prices (experiments/price_rule_check.py)

| price path vs official | after Singapore, MAE | moves exactly right |
|---|---|---|
| what the pipeline used before (rule fitted to rounds 0-2, our points) | $1.15m (worst $6.0m) | 39% |
| guide's default rule, official points | $0.17-0.19m | 77% of the path; 95-96% week-to-week |
| guide's default rule, our points | $0.85m | 77% week-to-week |

Learned from the official prices and now the defaults in `f1_prices.PriceRule`: floor $3.0m, cap $34.0m (two
observations), A tier from $18.5m, a driver who does not race keeps his price. Fitting the cut-offs to two real weekends
made the path much worse, so fitting now needs at least 6 real transitions.

## Game rules, checked against the official pages (fantasy.formula1.com game-rules and FAQ, read 2026-10-12)

What the solver (`f1_strategy.Rules`) assumes, and what the official FAQ says:

| rule | official | solver |
|---|---|---|
| free transfers | 2 after each race; one unused carries over to the next race only (max 3) | `free_per_round=2`, `max_bank=3` (same thing: next = min(3, left + 2)) |
| extra transfers | -10 points each | `penalty=10` |
| counting | net change against the previous race's line-up (reverting is free) | transfers = set difference between consecutive teams |
| chips | 6, each once a season, at most one per race, final once played | one chip per round, each at most once |
| Wildcard | unlimited changes within the cost cap | free transfers waived that round |
| Limitless | unlimited changes, no cost cap, team reverts afterwards; **no free transfers carry into the next race** | modelled as a parked week; the carry-over reset was added 2026-10-12 (before, the solver let up to 3 carry) |
| No Negative | any asset's negative score is limited to zero; transfer penalties still apply | E[max(X,0)] under the forecast error |
| Autopilot | the 2x boost goes to the highest-scoring driver | reported (expected gain), not optimised |
| Final Fix | one change between the end of Qualifying and the Grand Prix, no penalty | not modelled |
| team lock | Qualifying (Sprint weekends: the Sprint) | `docs/OPENF1_NOTES.md` |
