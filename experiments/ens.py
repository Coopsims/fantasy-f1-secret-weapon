"""Shared pieces for the ensemble experiments (phase 4 onwards)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from scipy.optimize import nnls
import mlflow_experiments as X

GBM_KW = dict(n_estimators=100, max_depth=2, learning_rate=0.03)      # selected on dev (2024) in phase 2


def wide_table(truth, fcs):
    w = truth[["t", "driver", "team", "total"]].copy()
    for name, fc in fcs.items():
        w = w.merge(fc.drivers[["t", "driver", "pred"]].rename(columns={"pred": name}), on=["t", "driver"], how="inner")
    return w


def rolling_nnls(wide, names, targets_t, min_rows=200):
    """Weights re-fit on ALL earlier weekends before each target; normalised to sum to 1."""
    rows, weights = [], []
    for t in targets_t:
        tr, te = wide[wide["t"] < t], wide[wide["t"] == t]
        if len(tr) < min_rows or te.empty:
            continue
        w, _ = nnls(tr[names].values, tr["total"].values)
        w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
        weights.append((t, *w))
        rows += [(t, d, float(v)) for d, v in zip(te["driver"], te[names].values @ w)]
    return (X.Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"])),
            pd.DataFrame(weights, columns=["t"] + names))


def adjust_constructors(sim_fc, mixed_fc, wide):
    """Constructor forecast = simulator's, shifted by the summed driver-level correction (mixed - sim)."""
    d = wide[["t", "driver", "team"]].merge(sim_fc.drivers[["t", "driver", "pred"]], on=["t", "driver"]) \
        .merge(mixed_fc.drivers[["t", "driver", "pred"]], on=["t", "driver"], suffixes=("_sim", "_mix"))
    d["adj"] = d["pred_mix"] - d["pred_sim"]
    adj = d.groupby(["t", "team"], as_index=False)["adj"].sum()
    c = sim_fc.constructors.merge(adj, on=["t", "team"], how="left")
    c["pred"] = c["pred"] + c["adj"].fillna(0.0)
    return X.Forecast(mixed_fc.drivers, c[["t", "team", "pred"]])


def subset(fc, ts):
    c = fc.constructors[fc.constructors["t"].isin(ts)] if fc.constructors is not None else None
    return X.Forecast(fc.drivers[fc.drivers["t"].isin(ts)].reset_index(drop=True), c)


def base_forecasts(res, truth, ctruth, targets):
    return {"sim": X.Sim().predict(res, truth, ctruth, targets),
            "gbm": X.GBM(**GBM_KW).predict(res, truth, ctruth, targets),
            "ewma": X.Baseline("ewma", 0.2).predict(res, truth, ctruth, targets),
            "last3": X.Baseline("last3").predict(res, truth, ctruth, targets)}
