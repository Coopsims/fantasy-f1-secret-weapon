# Overnight task log (2026-10-11 22:40 MDT -> 2026-10-12 06:00 MDT)

## Goal (from the user)
1. Best possible 2026 season predictor, hyperparameter tuned, everything in MLflow.
2. Season planner: any team (or none) + real prices + bank + free transfers + chips -> best lineups for the remaining races.
3. Next-round adjustments from an existing team under the game's constraints (plan_next_race.py).

## Ground rules
- Every prediction uses only earlier weekends (walk-forward). 2026 data may now be used for selection, but an honest
  number needs nested evaluation: tune on 2026 gamedays 1-9, score gamedays 10-17; then re-tune on all 17 for deployment.
- Primary truth for 2026: official game points (fantasy feed; own-car weekends only). Report reconstructed-point metrics too.
- Report only numbers produced; say when a gain is not confirmed.

## State at start
- Branch claude/magical-tesla-t5o3xe @ f543a97 (pushed). 98 tests pass.
- Data: openf1_full/ (87 weekends), telemetry/, fantasy feed (official points + prices for gamedays 1-18).
- Best so far: C_mix = adaptive blend of mean(tuned xgb, catboost, extratrees, rf on zoo 'all' features) and the simulator.
  2025 val RMSE 11.72 (vs sim -0.35, CI excl. 0; vs previous best -0.24, CI excl. 0); 2026 holdout -0.20 vs sim, CI incl. 0.
  Registered as f1-driver-points v1, alias 'candidate'.
- Season (official points/prices): optimum 5741 (proven); play policy ewma_blend 0.3-0.4 ~72% of optimum vs 61% without.

## Progress
(append below)
- 22:40 nested H1->H2 design aborted: sim best early season, trees best late -> blend schedule (b_early,b_late,kappa) + interleaved split (tune odd gamedays, test even). Relaunched 22:45 (250 trials x 5 families).
- 22:55 f1_planner.py + plan_season.py working end to end (future-race feature panels, simulator, price projection, exact MILP); Limitless chip added (frozen-round solver option + week enumeration); 101 tests pass.
- 23:02 season replay (decision test) built + verified on gamedays 1 and 10; baseline run (sim, sim+ewma0.4) launched.
- 23:04 found orphaned workers from the aborted search + stopped replay (killing a ProcessPool parent leaves workers alive) - killed them; load 77 -> falling. Rule: kill the whole process group, or pkill -P.
- 00:00 interleaved search done (74 min, 1250 trials; eval re-run after an MLflow metric-name crash). Held-out even gamedays: mean(ET,XGB) RMSE 14.59 vs sim 15.22 (-0.63 [-1.29,+0.01]), vs previous best 15.00; pick5 0.81 vs 0.71. Constructors: derived (c_sim 0, bonus alpha 0.2) RMSE 20.6 vs sim 22.2 on both halves. Provisional results/s26_final_spec.json written. Sim tuning on 2026 running.
- 00:01 sim tuning on 2026: no gain (best odd -0.07, even +0.08 vs default) -> default simulator kept. Deployment tuning launched: extratrees + xgb, 200 trials each, objective all 17 gamedays, warm-started from the odd-tuned bests.
