"""Does the Final Fix advisor's input help?  For every 2026 weekend: the simulator's Grand Prix points forecast before
the weekend vs conditioned on the real qualifying order (what final_fix.py uses), scored against the race points
actually scored (reconstructed, without Driver of the Day on both sides).  Logged to MLflow."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

if __name__ == "__main__":
    import mlflow_experiments as X
    from f1_weekend_model import WeekendForecaster
    res, truth, _ = X.get_data()
    f = WeekendForecaster(res, year=2026, verbose=False)
    tr = truth[truth["year"] == 2026]
    rows = []
    for rnd, g in tr.groupby("round"):
        g = g.dropna(subset=["quali_pos"])
        order = list(g.sort_values("quali_pos")["driver"])
        out = {}
        for kind, known in (("before", None), ("after_quali", {"quali": order})):
            fc = f.forecast(int(rnd), n=6000, seed=900 + int(rnd), known=known)
            out[kind] = dict(zip(fc.drivers, fc.event_points("race", include_dotd=False).mean(0)))
        m = g.assign(before=g["driver"].map(out["before"]), after=g["driver"].map(out["after_quali"])).dropna(subset=["before", "after"])
        r = {"round": int(rnd), "n": len(m)}
        for k in ("before", "after"):
            r[f"rmse_{k}"] = float(np.sqrt(((m[k] - m["race"]) ** 2).mean()))
            r[f"spearman_{k}"] = X._spearman(m[k], m["race"])
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    bt = X.paired_bootstrap(df.assign(t=df["round"], rmse=df["rmse_after"]), df.assign(t=df["round"], rmse=df["rmse_before"]), "rmse")
    print(df.mean().round(3).to_string())
    print(f"after-qualifying minus before: RMSE {bt['delta']:+.2f} [{bt['lo']:+.2f}, {bt['hi']:+.2f}]")
    df.to_csv("results/final_fix_check.csv", index=False)
    mm = X._mlflow()
    with mm.start_run(run_name=f"final-fix input check {datetime.date.today()}"):
        mm.set_tags({"phase": "final-fix-check", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                     "truth": "reconstructed race points, 2026, no DotD"})
        mm.log_metrics({"rmse_before": float(df.rmse_before.mean()), "rmse_after": float(df.rmse_after.mean()),
                        "spearman_before": float(df.spearman_before.mean()), "spearman_after": float(df.spearman_after.mean()),
                        "delta": bt["delta"], "lo": bt["lo"], "hi": bt["hi"]})
        mm.log_artifact("results/final_fix_check.csv")
