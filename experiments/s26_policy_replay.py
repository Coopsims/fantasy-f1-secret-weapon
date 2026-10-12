"""Re-play the 2026 season with cached walk-forward forecasts under different decision settings (chip reserve value,
discount).  In-sample on 2026; halves reported.  Logged to MLflow."""
import os, sys, glob, datetime, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
from experiments.s26_season_replay import play

if __name__ == "__main__":
    import mlflow_experiments as X
    path = sorted(glob.glob("results/cache/season_replay_forecasts_*.pkl"))[-1]
    res_ = pd.read_pickle(path)
    names = list(res_[0][2])
    rows = []
    for name, disc, cr in itertools.product(names, (0.6, 0.8, 1.0), (0.0, 10.0, 20.0, 30.0, 45.0)):
        fc = {g: (rounds, out[name]) for g, rounds, out in res_}
        tot, log = play(fc, discount=disc, chip_reserve=cr)
        chips = log[log.chip.notna() & (log.chip != "")][["gameday", "chip"]].values.tolist()
        rows.append({"forecaster": name, "discount": disc, "chip_reserve": cr, "points": tot, "share": tot / 5741.0,
                     "first_half": float(log.points.values[:8].sum()), "second_half": float(log.points.values[8:].sum()),
                     "chips_used": str(chips)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/s26_policy_replay.csv", index=False)
    mlflow = X._mlflow()
    with mlflow.start_run(run_name=f"s26-policy-replay {datetime.date.today()}"):
        mlflow.set_tags({"phase": "s26-policy", "data_version": X.DATA_VERSION, "forecasts": path, "note": "in-sample on 2026"})
        for r in rows:
            with mlflow.start_run(run_name=f"{r['forecaster']} disc {r['discount']} reserve {r['chip_reserve']}", nested=True):
                mlflow.log_params({k: r[k] for k in ("forecaster", "discount", "chip_reserve")})
                mlflow.log_metrics({k: r[k] for k in ("points", "share", "first_half", "second_half")})
        mlflow.log_artifact("results/s26_policy_replay.csv")
    print(df.pivot_table(index=["forecaster", "discount"], columns="chip_reserve", values="points").to_string())
