"""One-lever sweep of the play policy's forecast shrinkage (ewma_blend), everything else fixed, on the real 2026 season.
Reports total and per-half gaps to the hindsight optimum, and logs to MLflow. Uses the cached replay from `f1_season.py tune`."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import f1_season as S
from f1_strategy import Rules
from f1_weekend_data import load_results
import mlflow_experiments as X

if __name__ == "__main__":
    res = load_results(verbose=False)
    rules = Rules()
    rep = S.cached_replay(res, 2026, 4000, 4, cache="season_2026_replay.pkl")
    act = rep["actuals"]
    prices, n_est, rule = S.price_table(act["rounds"], act["pts"], "fantasy_f1_prices.csv", return_rule=True)
    plan, _ = S.hindsight_optimum(act, prices, rules)
    print(f"hindsight optimum {plan.total:.0f}; proven optimal: {plan.optimal} ({plan.status.strip()})")
    base = dict(horizon=3, discount=0.6, price_aware=True, value_weight=0.0, chip_cost=0.0)
    grid = [S.PlayPolicy(**base, ewma_blend=w) for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)]
    df = S.tune_play(rep, prices, rules, rule, grid, workers=8, optimum=plan, verbose=False).sort_values("ewma_blend")
    half = len(plan.round_points) // 2
    print(df[["ewma_blend", "points", "delta", "delta_first_half", "delta_second_half"]].to_string(index=False))
    mlflow = X._mlflow()
    with mlflow.start_run(run_name=f"season-2026 ewma_blend sweep {datetime.date.today()}"):
        mlflow.set_tags({"phase": "season-policy", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                         "note": "in-sample on the 2026 season (the only season with prices); halves reported separately",
                         "optimum_proven": str(plan.optimal), "prices_estimated_weekends": str(n_est)})
        mlflow.log_metric("hindsight_optimum", plan.total)
        for r in df.itertuples():
            with mlflow.start_run(run_name=f"ewma_blend={r.ewma_blend}", nested=True):
                mlflow.log_params({**base, "ewma_blend": r.ewma_blend})
                mlflow.log_metrics({"points": r.points, "gap": r.delta, "gap_first_half": r.delta_first_half,
                                    "gap_second_half": r.delta_second_half, "share_of_optimum": r.points / plan.total})
