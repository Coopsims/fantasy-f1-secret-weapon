"""Phase 4: ensembles. All base forecasts are walk-forward over one continuous pass (targets <= 2025), so every
training row used by a stacker was itself produced without seeing its own weekend or later.
  blend3   fixed weights on (sim, gbm, ewma), grid searched on dev (2024)
  nnls     weights re-fit on all earlier weekends before each target (adaptive, no tuning)
  stackgbm GBM that also sees the simulator's forecast, spread and DNF probability (direct and residual targets)
"""
import os, sys, datetime, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from experiments import ens
import mlflow_experiments as X

SPLIT_YEAR = X.SPLITS


def subset(fc, ts):
    c = fc.constructors[fc.constructors["t"].isin(ts)] if fc.constructors is not None else None
    return X.Forecast(fc.drivers[fc.drivers["t"].isin(ts)].reset_index(drop=True), c)


def wide_table(truth, fcs):
    w = truth[["t", "driver", "total"]].copy()
    for name, fc in fcs.items():
        w = w.merge(fc.drivers[["t", "driver", "pred"]].rename(columns={"pred": name}), on=["t", "driver"], how="inner")
    return w


def rolling_nnls(wide, names, targets_t, min_rows=200):
    rows = []
    for t in targets_t:
        tr, te = wide[wide["t"] < t], wide[wide["t"] == t]
        if len(tr) < min_rows or te.empty:
            continue
        w, _ = nnls(tr[names].values, tr["total"].values)
        w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
        rows += [(t, d, float(v)) for d, v in zip(te["driver"], te[names].values @ w)]
    return X.Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"]))


class StackGBM(X.GBM):
    kind = "stack_gbm"
    def __init__(self, sim_fc, residual=False, **kw):
        super().__init__(**kw); self.sim, self.residual = sim_fc, residual
    def describe(self):
        return {**super().describe(), "residual_target": self.residual, "extra_features": "sim_pred,sim_spread,sim_p_dnf"}
    def predict(self, res, truth, ctruth, targets):
        from xgboost import XGBRegressor
        feats = X.GBM_FEATURES + ["sim_pred", "sim_spread", "sim_p_dnf"]
        s = self.sim.drivers.assign(sim_spread=lambda d: d["p90"] - d["p10"])[["t", "driver", "pred", "sim_spread", "p_dnf"]]
        s = s.rename(columns={"pred": "sim_pred", "p_dnf": "sim_p_dnf"})
        panel = X._panel_features(truth).merge(s, on=["t", "driver"], how="inner")
        rows = []
        for w in targets.itertuples():
            t = int(w.t)
            tr = panel[panel["t"] < t].dropna(subset=["d_last1"]); te = panel[panel["t"] == t]
            if len(tr) < self.min_rows or te.empty:
                continue
            y = tr["total"] - tr["sim_pred"] if self.residual else tr["total"]
            m = XGBRegressor(**self.kw).fit(tr[feats], y)
            pr = m.predict(te[feats]) + (te["sim_pred"].values if self.residual else 0)
            rows += [(t, d, float(v)) for d, v in zip(te["driver"], pr)]
        return X.Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"]))


if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    wk = res.weekends[res.weekends["year"] <= 2025]
    allt = wk[[(res.weekends["t"] < t).sum() >= 8 for t in wk["t"]]].reset_index(drop=True)
    gbm_kw = dict(ens.GBM_KW)                                                  # selected on dev in phase 2
    base = {"sim": X.Sim().predict(res, truth, ctruth, allt),
            "gbm": X.GBM(**gbm_kw).predict(res, truth, ctruth, allt),
            "ewma": X.Baseline("ewma", 0.2).predict(res, truth, ctruth, allt)}
    ts = {s: set(allt[allt["year"] == y]["t"]) for s, y in SPLIT_YEAR.items() if s != "test"}
    cands = {}
    # fixed 3-way blend, weights chosen on dev
    best, grid = None, [(a, b) for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01, 0.1) if a + b <= 1.0001]
    wide = wide_table(truth, base)
    dev = wide[wide["t"].isin(ts["dev"])]
    scored = sorted(((float((np.abs(a * dev["sim"] + b * dev["gbm"] + (1 - a - b) * dev["ewma"] - dev["total"])).mean()), round(a, 1), round(b, 1)) for a, b in grid))
    mae, wa, wb = scored[0]
    print(f"blend3 dev-selected weights sim={wa} gbm={wb} ewma={round(1 - wa - wb, 1)} (dev MAE {mae:.3f})")
    f3 = X.Forecast(wide.assign(pred=wa * wide["sim"] + wb * wide["gbm"] + (1 - wa - wb) * wide["ewma"])[["t", "driver", "pred"]])
    cands["blend3 (dev weights)"] = f3
    cands["blend2 sim0.4+gbm"] = X.blend(base["sim"], base["gbm"], 0.4)
    cands["nnls(sim,gbm,ewma)"] = rolling_nnls(wide, ["sim", "gbm", "ewma"], allt["t"])
    cands["nnls(sim,gbm)"] = rolling_nnls(wide, ["sim", "gbm"], allt["t"])
    cands["stack_gbm direct"] = StackGBM(base["sim"], False, **gbm_kw).predict(res, truth, ctruth, allt)
    cands["stack_gbm residual"] = StackGBM(base["sim"], True, **gbm_kw).predict(res, truth, ctruth, allt)
    with mlflow.start_run(run_name=f"phase4-stacking {datetime.date.today()}"):
        mlflow.set_tags({"phase": "4-stacking", "git_sha": X.git_sha(), "blend3_weights": f"sim={wa},gbm={wb}"})
        for split in ("dev", "val"):
            ref = None
            for name, fc in [("sim_default", base["sim"]), ("gbm_selected", base["gbm"]), ("ewma_0.2", base["ewma"])] + list(cands.items()):
                sub = subset(fc, ts[split])
                model = X.Sim() if name == "sim_default" else (X.GBM(**gbm_kw) if name.startswith("gbm") else X.Baseline("ewma", 0.2) if name.startswith("ewma") else X.Baseline("ensemble"))
                _, pw, s = X.run_and_log(f"{name} [{split}]", model, split, ref=ref, tags={"phase": "4", "cand": name}, fc_override=sub)
                if name == "sim_default":
                    ref = ("sim_default", pw)
                bt = X.paired_bootstrap(pw, ref[1], "mae")
                print(f"{split} {name:24s} MAE {s['mae']:.3f} rho {s['spearman']:.3f} top5 {s['top5']:.3f} | vs sim {bt['delta']:+.3f} [{bt['lo']:+.3f},{bt['hi']:+.3f}]")
