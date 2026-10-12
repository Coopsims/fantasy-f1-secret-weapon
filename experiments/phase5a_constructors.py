"""Phase 5a: does the driver-level ensemble also improve constructor forecasts? (dev/val only)"""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments import ens
import mlflow_experiments as X

if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    wk = res.weekends[res.weekends["year"] <= 2025]
    allt = wk[[(res.weekends["t"] < t).sum() >= 8 for t in wk["t"]]].reset_index(drop=True)
    base = ens.base_forecasts(res, truth, ctruth, allt)
    wide = ens.wide_table(truth, base)
    nn, wts = ens.rolling_nnls(wide, ["sim", "gbm"], allt["t"])
    fixed = X.blend(base["sim"], base["gbm"], 0.4)
    # naive constructor baseline: mean of the team's last 3 weekend totals
    ct = ctruth.sort_values("t")
    rows = []
    for t in allt["t"]:
        for team in ct[ct["t"] == t]["team"]:
            h = ct[(ct["team"] == team) & (ct["t"] < t)]["total"].tail(3)
            rows.append((t, team, float(h.mean()) if len(h) else float(ct[ct["t"] < t]["total"].tail(30).mean())))
    cnaive = X.Forecast(base["sim"].drivers, __import__("pandas").DataFrame(rows, columns=["t", "team", "pred"]))
    cands = {"sim_default": base["sim"], "nnls(sim,gbm)": ens.adjust_constructors(base["sim"], nn, wide),
             "blend 0.4 sim + gbm": ens.adjust_constructors(base["sim"], fixed, wide), "constructor naive last3": cnaive}
    with mlflow.start_run(run_name=f"phase5a-constructors {datetime.date.today()}"):
        mlflow.set_tags({"phase": "5a-constructors", "git_sha": X.git_sha()})
        for split, y in (("dev", 2024), ("val", 2025)):
            ts = set(allt[allt["year"] == y]["t"])
            ref = None
            for name, fc in cands.items():
                sub = ens.subset(fc, ts)
                _, pw, s = X.run_and_log(f"constructors: {name} [{split}]", X.Sim(), split, ref=ref, tags={"phase": "5a", "cand": name}, fc_override=sub)
                if name == "sim_default":
                    ref = ("sim_default", pw)
                bt = X.paired_bootstrap(pw.rename(columns={"c_mae": "cm"})[["t", "cm"]], ref[1].rename(columns={"c_mae": "cm"})[["t", "cm"]], "cm")
                print(f"{split} {name:26s} driver MAE {s['mae']:.3f} | constructor MAE {s['c_mae']:.3f} | vs sim {bt['delta']:+.3f} [{bt['lo']:+.3f},{bt['hi']:+.3f}]")
