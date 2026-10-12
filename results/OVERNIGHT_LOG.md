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
- 00:03 other session pushed d0179e5 (pit points in f1_season.play_season on official points; policy-set labels) - rebased onto it, tests pass, pushed. My s26 replay forecasts already include pit points via simulator / official bonus.
- 00:06 season replay (official, rolling 3): provisional ET+XGB 3971 (69%), sim+ewma0.4 3955, sim 3330; policy variants +-50 (noise). Forecast accuracy even gamedays: new 14.59 vs sim+ewma0.4 15.12. Draft report results/BEST_MODEL_2026.md. Deploy tuning running.
- 00:10 LEAK FOUND + FIXED: team affinity features (aff_team*) included the teammate's same-weekend row (zoo._affinity/_sim_affinity used 'earlier in sort order', not 'strictly earlier t'); found by a future-panel vs panel consistency check. Test added. Panel rebuilt (panel_v2): only those 3 features changed. Held-out (even) re-scored: ET 14.65 (unaffected, no_track), XGB 14.69->14.75, ET+XGB 14.62, sim 15.22. Stale planner caches moved out. Deploy tuning relaunched on fixed data. Phase 7 (2024-25) results used the leaky features for 'all' sets - flagged.
- 00:17 phase 7a re-run on the fixed panel: 'all' feature sets were inflated by ~0.2 RMSE (2024-25); extratrees/all now dev 10.59 val 11.84 (still ahead of previous best 10.77/11.96 by ~0.12-0.19). Registered models f1-driver-points v1 and f1-fantasy-points-2026 v1 tagged superseded (leaky panel).
- 00:21 decision test on the fixed panel: new 3953 = sim+ewma 3955 (tie), sim 3330. Decision variants (cached forecasts): avg(new, sim+ewma0.4) 4151 (72%) best; new+ewma0.2 4035. avg costs +0.16 RMSE (CI incl. 0). Planners now use a planning blend 0.5 model + 0.3 sim + 0.2 recent form (--sharp disables) - optimiser's-curse shrinkage; evidence one season, in-sample.
- 00:34 deploy tuning (fixed panel, all 17 gamedays): xgb 14.45, ET 14.64, mean 14.47 (in-sample). Final spec written; registered f1-fantasy-points-2026 v3 alias champion (v2 = mistake with the provisional spec, tagged superseded).
- 00:37 final spec + champion v3 + SEASON_PLAN.md + report committed. Austin: OUT Hamilton, Colapinto, Racing Bulls / IN Mercedes, Hulkenberg, Lindblad, 2x Antonelli (214.5 vs 181.6). Next: value No Negative / Autopilot with the forecast error distribution.
- 00:41 No Negative valued with forecast error (+12 alone), Autopilot advice (~6/week); committed. Next: seed-averaging test.
- 00:51 seed averaging (5 seeds/family, even gamedays): 14.621 vs 14.624 single (-0.001 [-0.027,+0.022]); seed sd 0.3 pts, top-5 overlap 4.82/5 -> deployed per the pre-set rule (removes seed luck). Champion v4 (v3 tagged superseded). Austin unchanged (212.4 vs 181.4).
- 00:51 game rules checked on the official FAQ: all solver rules match except Limitless (no free transfers carry out of a Limitless week) - fixed in solver + evaluate_plan, test added. Scenarios: none 994, 3x 1024, NN 1006, WC 996, Limitless 1050 (now Mexico City), all 1091.
