"""Phase 5c: EXPLORATORY, post-hoc look at where the 2026 ensemble gain/loss comes from.
Not used to select, tune or change anything (the holdout in phase 5b is already fixed and reported)."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from experiments import ens
import mlflow_experiments as X

if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    wk = res.weekends
    allt = wk[[(wk["t"] < t).sum() >= 8 for t in wk["t"]]].reset_index(drop=True)
    base = ens.base_forecasts(res, truth, ctruth, allt)
    wide = ens.wide_table(truth, base)
    nn, wts = ens.rolling_nnls(wide, ["sim", "gbm"], allt["t"])
    prim = ens.adjust_constructors(base["sim"], nn, wide)
    t26 = list(allt[allt["year"] == 2026]["t"])
    pws = {n: X.per_weekend(ens.subset(f, set(t26)), truth, ctruth) for n, f in (("sim", base["sim"]), ("primary", prim), ("gbm", base["gbm"]))}
    d = pws["sim"][["round", "location", "mae", "c_mae"]].rename(columns={"mae": "sim_mae", "c_mae": "sim_c"}) \
        .merge(pws["primary"][["round", "mae", "c_mae"]].rename(columns={"mae": "ens_mae", "c_mae": "ens_c"}), on="round") \
        .merge(pws["gbm"][["round", "mae"]].rename(columns={"mae": "gbm_mae"}), on="round")
    d["d_driver"] = d.ens_mae - d.sim_mae
    d["d_constr"] = d.ens_c - d.sim_c
    print(d.round(2).to_string(index=False))
    for name, sl in (("first 4 weekends of 2026", d.iloc[:4]), ("remaining 13", d.iloc[4:])):
        print(f"{name:26s} driver MAE sim {sl.sim_mae.mean():.2f} ens {sl.ens_mae.mean():.2f} gbm {sl.gbm_mae.mean():.2f} | constructor sim {sl.sim_c.mean():.2f} ens {sl.ens_c.mean():.2f}")
    with mlflow.start_run(run_name=f"phase5c-exploratory-2026 {datetime.date.today()}"):
        mlflow.set_tags({"phase": "5c-exploratory", "note": "post-hoc; not used for selection", "git_sha": X.git_sha()})
        for r in d.itertuples():
            mlflow.log_metrics({"d_driver_mae": r.d_driver, "d_constr_mae": r.d_constr}, step=int(r.round))
