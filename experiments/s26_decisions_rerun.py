"""Re-score the 2026 decision tests after the rule fixes from the game's FAQ (x3 Boost goes on a second driver while the
x2 stays; no free transfers carry out of a Limitless week): the proven hindsight optimum on official points and prices,
and the walk-forward season play with the cached forecasts (results/cache/season_replay_forecasts_*.pkl - forecasts are
unchanged, only the decisions and their scoring).  Logged to MLflow."""
import os, sys, glob, datetime, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

OPT_TIME = int(os.environ.get("OPT_TIME", "1800"))

if __name__ == "__main__":
    import mlflow_experiments as X
    import f1_season as S
    from f1_strategy import Rules
    from f1_weekend_data import load_results
    from experiments.s26_season_replay import play
    path = sorted(glob.glob("results/cache/season_replay_forecasts_*.pkl"))[-1]
    res_ = pd.read_pickle(path)
    fc = {name: {g: (rounds, out[name]) for g, rounds, out in res_} for name in res_[0][2]}
    model = next(n for n in fc if n not in ("sim", "sim+ewma0.4"))
    fc["planning blend (0.5 model + 0.5 sim+ewma0.4)"] = {g: (r, 0.5 * fc[model][g][1] + 0.5 * fc["sim+ewma0.4"][g][1]) for g, (r, _) in fc[model].items()}
    rows = []
    for name, f in fc.items():
        tot, log = play(f)
        rows.append({"forecaster": name, "season_points": tot, "first_half": float(log.points.values[:8].sum()),
                     "second_half": float(log.points.values[8:].sum()),
                     "chips": str(log[log.chip.notna() & (log.chip != "")][["gameday", "chip"]].values.tolist())})
        print(rows[-1], flush=True)
    res = load_results(verbose=False)
    rules = Rules()
    official = pd.read_csv("fantasy_f1_official_points.csv")
    rep = S.cached_replay(res, 2026, 4000, 4, official=official, cache="season_2026_replay_official.pkl")
    act = rep["actuals"]
    prices, n_est, rule = S.price_table(act["rounds"], act["pts"], "fantasy_f1_prices_official.csv", verbose=False,
                                        return_rule=True, active=act.get("active"))
    t0 = time.time()
    plan, opt = S.hindsight_optimum(act, prices, rules, time_limit=OPT_TIME)
    print(f"hindsight optimum {plan.total:.0f}; proven optimal: {plan.optimal} ({plan.status.strip()}); "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)
    df = pd.DataFrame(rows)
    df["share_of_optimum"] = df["season_points"] / plan.total
    df = df.sort_values("season_points", ascending=False)
    df.to_csv("results/s26_decisions_rerun.csv", index=False)
    opt.to_csv("results/season_2026_official_optimum.csv", index=False)
    m = X._mlflow()
    with m.start_run(run_name=f"s26 decisions re-run (FAQ rules) {datetime.date.today()}"):
        m.set_tags({"phase": "s26-decisions-rerun", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "forecasts": path,
                    "rules": "x3 on a second driver + x2 kept; Limitless resets free transfers", "optimum_proven": str(plan.optimal)})
        m.log_metric("optimum", float(plan.total))
        for r in df.to_dict("records"):
            with m.start_run(run_name=f"season play: {r['forecaster']}", nested=True):
                m.set_tags({"phase": "s26-decisions-rerun", "forecaster": r["forecaster"], "chips": r["chips"]})
                m.log_metrics({k: float(r[k]) for k in ("season_points", "share_of_optimum", "first_half", "second_half")})
        m.log_artifact("results/s26_decisions_rerun.csv")
    print(df.round(3).to_string(index=False))
