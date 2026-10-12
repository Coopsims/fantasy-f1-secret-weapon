"""The 2026 season on the game's OFFICIAL points and prices (fantasy feed): replay accuracy, hindsight optimum
(longer solver limit), default play, the 128-policy search and the forecast-shrinkage sweep - one optimum solve,
everything logged to MLflow. In-sample on 2026 (the only season with prices); halves reported separately."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import f1_season as S
from f1_strategy import Rules
from f1_weekend_data import load_results
import mlflow_experiments as X

OPT_TIME = int(os.environ.get("OPT_TIME", "2400"))

if __name__ == "__main__":
    res = load_results(verbose=False)
    rules = Rules()
    official = pd.read_csv("fantasy_f1_official_points.csv")
    rep = S.cached_replay(res, 2026, 4000, 4, official=official, cache="season_2026_replay_official.pkl")
    acc, act = rep["accuracy"], rep["actuals"]
    prices, n_est, rule = S.price_table(act["rounds"], act["pts"], "fantasy_f1_prices_official.csv", verbose=True,
                                        return_rule=True, active=act.get("active"))
    print(f"replay vs official points: driver MAE {acc.driver_mae.mean():.2f} (naive {acc.driver_mae_naive3.mean():.2f}), "
          f"rank-corr {acc.driver_rankcorr.mean():.2f}, cover {acc.driver_cover80.mean():.2f}; constructor MAE "
          f"{acc.constructor_mae.mean():.2f} (naive {acc.constructor_mae_naive3.mean():.2f}); estimated price weekends: {n_est}")
    plan, opt = S.hindsight_optimum(act, prices, rules, time_limit=OPT_TIME)
    print(f"hindsight optimum {plan.total:.0f}; proven optimal: {plan.optimal} ({plan.status.strip()}); objective bound {plan.objective}")
    default = S.play_season(rep, prices, rules, verbose=False)
    grid = S.default_policy_grid()
    base = dict(horizon=3, discount=0.6, price_aware=True, value_weight=0.0, chip_cost=0.0)
    sweep = [S.PlayPolicy(**base, ewma_blend=w) for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)]
    df = S.tune_play(rep, prices, rules, rule, grid + sweep, workers=8, optimum=plan, verbose=False)
    # tune_play returns rows sorted by gap: label them by their position in the input list, not by row order
    df["set"] = np.where(df["grid_index"] < len(grid), "grid", "ewma_sweep")
    df.to_csv("results/season_2026_official_policies.csv", index=False)
    opt.to_csv("results/season_2026_official_optimum.csv", index=False)
    acc.to_csv("results/season_2026_official_accuracy.csv", index=False)
    g = df[df.set == "grid"]
    hist = pd.read_csv("fantasy_f1_history.csv")
    print(f"\nseason totals (17 weekends): optimum {plan.total:.0f} | default play {default['total']:.0f} "
          f"({default['total'] / plan.total:.0%}) | best of 128 {g.points.max():.0f} | median of 128 {g.points.median():.0f} | yours (2 rounds recorded) {hist.actual_points.sum():.0f}")
    for c in ("horizon", "discount", "price_aware", "value_weight", "chip_cost", "ewma_blend"):
        print(f"  mean points by {c:13s}", g.groupby(c)["points"].mean().round(0).to_dict())
    for sel, ev in (("delta_first_half", "delta_second_half"), ("delta_second_half", "delta_first_half")):
        pick = g.sort_values(sel).iloc[0]
        print(f"  pick on {sel[6:]}, score on {ev[6:]}: gap {pick[ev]:.0f} vs median {g[ev].median():.0f} ({(g[ev] < pick[ev]).mean():.0%} better)")
    print("\newma sweep:\n", df[df.set == "ewma_sweep"][["ewma_blend", "points", "delta", "delta_first_half", "delta_second_half"]].to_string(index=False))
    mlflow = X._mlflow()
    with mlflow.start_run(run_name=f"season-2026 official data {datetime.date.today()}"):
        mlflow.set_tags({"phase": "season-official", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                         "points": "official fantasy feed", "prices": "official fantasy feed",
                         "optimum_proven": str(plan.optimal), "note": "policy search in-sample on 2026"})
        mlflow.log_metrics({"optimum": plan.total, "default_play": default["total"], "best_policy": float(g.points.max()),
                            "median_policy": float(g.points.median()), "replay_driver_mae": float(acc.driver_mae.mean()),
                            "replay_constructor_mae": float(acc.constructor_mae.mean())})
        mlflow.log_artifact("results/season_2026_official_policies.csv")
