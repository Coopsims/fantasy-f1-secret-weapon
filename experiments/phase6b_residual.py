"""Phase 6b: constrained use of pre-lock information - ridge regression of (actual - simulator forecast) on practice /
Sprint Qualifying features, re-fit on earlier weekends only. Alpha chosen on dev (2024), confirmed on val (2025)."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from experiments import ens
import mlflow_experiments as X
import f1_practice as P

FEATS = ["fp_gap_min", "fp_gap_last", "fp_lr_gap", "fp_vs_mate", "fp_team_gap_min", "fp_rank_mean", "sq_rel"]


class ResidualRidge:
    kind = "residual_ridge"
    def __init__(self, sim_fc, ptab, alpha, cols=FEATS, extra_sim_terms=True):
        self.sim, self.ptab, self.alpha, self.cols, self.terms = sim_fc, ptab, alpha, cols, extra_sim_terms
    def describe(self):
        return {"alpha": self.alpha, "features": ",".join(self.cols), "sim_terms": self.terms}
    def _frame(self, truth):
        d = self.sim.drivers[["t", "driver", "pred"]].merge(truth[["t", "driver", "total", "team", "is_sprint"]], on=["t", "driver"]) \
            .merge(self.ptab[["t", "driver"] + self.cols], on=["t", "driver"], how="left")
        d["resid"] = d["total"] - d["pred"]
        return d
    def _design(self, d, stats=None):
        X_ = d[self.cols].copy()
        for c in self.cols:                              # clip heavy tails, standardise with TRAIN statistics only
            X_[c] = X_[c].clip(upper=X_[c].quantile(0.98) if stats is None else stats[c][2])
        if stats is None:
            stats = {c: (X_[c].mean(), X_[c].std() or 1.0, d[c].quantile(0.98)) for c in self.cols}
        Z = pd.DataFrame({c: (X_[c] - stats[c][0]) / stats[c][1] for c in self.cols})
        miss = Z.isna().astype(float).add_suffix("_na")
        Z = Z.fillna(0.0)
        if self.terms:
            Z["sim_pred"] = (d["pred"] - 12.0) / 12.0
            Z["sim_x_sprint"] = Z["sim_pred"] * d["is_sprint"].astype(float)
        return pd.concat([Z, miss[[c for c in miss if miss[c].std() > 0 or True]]], axis=1), stats
    def predict(self, res, truth, ctruth, targets):
        d = self._frame(truth)
        rows = []
        for w in targets.itertuples():
            t = int(w.t)
            tr, te = d[d["t"] < t], d[d["t"] == t]
            if len(tr) < 200 or te.empty:
                continue
            A, stats = self._design(tr)
            m = Ridge(alpha=self.alpha).fit(A, tr["resid"])
            B, _ = self._design(te, stats)
            rows += [(t, dr, float(p + c)) for dr, p, c in zip(te["driver"], te["pred"], m.predict(B))]
        return X.Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"]))


if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    ptab = P.build_practice_features(P.load_practice(), res.weekends, res.sessions)
    wk = res.weekends[res.weekends["year"] <= 2025]
    allt = wk[[(res.weekends["t"] < t).sum() >= 8 for t in wk["t"]]].reset_index(drop=True)
    ts = {"dev": set(allt[allt["year"] == 2024]["t"]), "val": set(allt[allt["year"] == 2025]["t"])}
    sim = X.Sim().predict(res, truth, ctruth, allt)
    gbm = X.GBM(**ens.GBM_KW).predict(res, truth, ctruth, allt)
    wide = ens.wide_table(truth, {"sim": sim, "gbm": gbm})
    prev, _ = ens.rolling_nnls(wide, ["sim", "gbm"], allt["t"])
    cands = {"sim_default": sim, "nnls(sim,gbm) [prev best]": prev}
    for alpha in (30, 100, 300, 1000, 3000):
        cands[f"sim + ridge(FP,SQ) a={alpha}"] = ResidualRidge(sim, ptab, alpha).predict(res, truth, ctruth, allt)
    for alpha in (300, 1000):
        cands[f"sim + ridge(sim terms only) a={alpha}"] = ResidualRidge(sim, ptab, alpha, cols=["sq_rel"]).predict(res, truth, ctruth, allt)
    # ensemble of the best ridge-corrected sim with the GBM
    with mlflow.start_run(run_name=f"phase6b-residual {datetime.date.today()}"):
        mlflow.set_tags({"phase": "6b-residual", "git_sha": X.git_sha()})
        out = {}
        for split in ("dev", "val"):
            ref = pr = None
            for name, fc in cands.items():
                sub = ens.subset(fc, ts[split])
                _, pw, s = X.run_and_log(f"{name} [{split}]", X.Sim(), split, ref=ref, tags={"phase": "6b", "cand": name}, fc_override=sub)
                if name == "sim_default": ref = ("sim_default", pw)
                if name.startswith("nnls"): pr = pw
                bt = X.paired_bootstrap(pw, ref[1], "mae"); vp = X.paired_bootstrap(pw, pr, "mae") if pr is not None and not name.startswith("nnls") and name != "sim_default" else None
                out[(split, name)] = s["mae"]
                print(f"{split} {name:36s} MAE {s['mae']:.3f} rho {s['spearman']:.3f} | vs sim {bt['delta']:+.3f} [{bt['lo']:+.3f},{bt['hi']:+.3f}]"
                      + (f" | vs prev {vp['delta']:+.3f} [{vp['lo']:+.3f},{vp['hi']:+.3f}]" if vp else ""))
