# Data correction (found 2026-10-11 while verifying the one-pull cache)

## What was wrong

1. **Sprint weekends in 2023 and 2024: Qualifying and Sprint Qualifying were swapped.**
   `_relabel_sessions` told the two "Qualifying" sessions of a Sprint weekend apart by guessing that the lower `session_key` is
   the Sprint one. OpenF1 `session_key` order is not chronological. Counting all 24 Sprint weekends in the new cache:

   | season | lower-key "Qualifying" session is really | guess right? |
   |---|---|---|
   | 2023 | Qualifying | no (6 weekends) |
   | 2024 | Qualifying | no (6 weekends) |
   | 2025 | Sprint Qualifying | yes |
   | 2026 | Sprint Qualifying | yes |

   (The Race/Sprint pair was right in every season.) In the affected weekends the pipeline treated the Sprint Qualifying
   result as "Qualifying" and the real Qualifying as "Sprint Qualifying": wrong qualifying points, wrong race grids, wrong
   inputs to the qualifying and race rating fits. **2025 and 2026 actual points were not affected**; the 2023-24 history
   the model learns from, and the 2024 (dev) evaluation targets, were.
   Fix: `session_name` is authoritative when the cache has it; the guess remains only for old caches without names.
   Check: with names, the position-derived order matches OpenF1's official `session_result` for all 222 sessions compared
   (pole/winner 100%; qualifying positions 99-100% exact; race positions 93-99%, differences mostly post-race penalties).

2. **2023 "source gaps" were failed requests.** Spielberg, Singapore and São Paulo 2023 had no positions/laps in the old cache and
   I reported them as missing at OpenF1. They are not: the one-pull cache has them. (Only 2023 Imola, which was cancelled, has no data.)
   2023 now has 22 weekends instead of 19.

3. **Consequence for phase 6 (practice features): a leak.** For 2023-24 Sprint weekends the `sq_pos` / `sq_rel` features and the
   simulator's "known Sprint Qualifying" input actually held the real Qualifying result, which happens after the team lock.
   Those phase-6 results were invalid and have been redone.

## What was redone

The raw caches were rebuilt from `openf1_full/` (`rebuild_raw_from_full.py`); all 311 earlier MLflow runs are kept but tagged
`superseded=true, data_version=v1-mislabelled-sprint-quali`; every phase (`experiments/run_all.py`), the 2026 season replay and the
2026 holdout were re-run with `data_version=v2-session-names`. The 2026 holdout was therefore scored a second time; the reason
is the data fix only: same candidate set, same selection procedure, same decision rule (stated in a new pre-registration run).

## What changed in the numbers

* Simulator, driver MAE: dev 8.05 -> 8.05, val 9.35 -> 9.36 (essentially unchanged); 2026 holdout 11.00 -> 11.02.
* Ensemble (simulator + gradient boosting) gain on the simulator: dev -0.20 -> -0.14 (95% interval now includes 0), val -0.21 -> -0.16 (still excludes 0).
  The direction holds; it is smaller and, on 2024, no longer individually significant.
* 2026 holdout, ensemble: -0.18 [-0.37, +0.02] -> -0.18 [-0.34, +0.00]: still not confirmed by the pre-registered rule; constructors still worse (+0.34).
* Practice / Sprint Qualifying features: before, "no improvement"; now a small improvement that is not confirmed (see MODEL_EXPERIMENTS.md).

## Not fixed

Race finishing order comes from the last `position` record, not the official classification, and the DNF flag is a
"completed under 90% of the longest runner's laps" rule. `session_result` (official positions plus dnf/dns/dsq flags) is now in
the cache and would be the better source, but changing the actuals is a larger change that I have not made.
