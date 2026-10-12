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
    fcs = {name: {g: (rounds, out[name]) for g, rounds, out in res_} for name in res_[0][2]}
    model = next(n for n in fcs if n not in ("sim", "sim+ewma0.4"))
    fcs["planning blend"] = {g: (r, 0.5 * fcs[model][g][1] + 0.5 * fcs["sim+ewma0.4"][g][1]) for g, (r, _) in fcs[model].items()}
    names = list(fcs)
    rows = []
    reserves = [float(x) for x in os.environ.get("RESERVES", "0,10,20,30,45").split(",")]
    for name, disc, cr in itertools.product(names, (0.6, 0.8, 1.0), reserves):
        fc = fcs[name]
        tot, log = play(fc, discount=disc, chip_reserve=cr)
        chips = log[log.chip.notna() & (log.chip != "")][["gameday", "chip"]].values.tolist()
        rows.append({"forecaster": name, "discount": disc, "chip_reserve": cr, "points": tot, "share": tot / 5778.0,
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
