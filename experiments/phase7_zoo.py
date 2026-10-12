"""Phase 7a: sweep of model families x feature sets on dev (2024) and val (2025), every config logged to MLflow with
its fitted model (the last walk-forward refit of the split).  References: simulator, EWMA, previous best ensemble."""
import os, sys, datetime, pickle, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

FAMILIES = ["ridge", "elasticnet", "huber", "knn", "svr", "mlp", "rf", "extratrees", "hgb", "xgb", "lgbm", "catboost"]
FSETS = ["base", "no_track", "no_weekend", "all"]
PANEL = os.environ.get("PANEL_PKL", "")


def _panel():
    import mlflow_experiments as X, f1_practice as P, openf1_cache as C
    from experiments import zoo
    res, truth, ctruth = X.get_data()
    if PANEL and os.path.exists(PANEL):
        return res, truth, ctruth, pd.read_pickle(PANEL)
    return res, truth, ctruth, zoo.build_panel(res, truth, full=C.load_all(), pdata=P.load_practice(), tracks=zoo.track_tables())


def _job(args):
    family, fset, split, ts = args
    from experiments import zoo
    res, truth, ctruth, panel = _panel_cached()
    feats = zoo.FEATURES[fset]
    t0 = time.time()
    fc = zoo.walk_forward(panel, feats, family, None, ts)
    secs = time.time() - t0
    tr = panel[(panel["t"] < max(ts)) & panel["d_last1"].notna()]
    model = zoo.make_model(family).fit(tr[feats], tr["total"])        # = the last walk-forward refit's training set
    return family, fset, split, fc, secs, pickle.dumps(model)


_CACHE = {}
def _panel_cached():
    if "p" not in _CACHE:
        _CACHE["p"] = _panel()
    return _CACHE["p"]


if __name__ == "__main__":
    import mlflow_experiments as X
    from experiments import zoo, ens
    res, truth, ctruth, panel = _panel_cached()
    if not PANEL:
        os.environ["PANEL_PKL"] = PANEL_PATH = os.path.join(tempfile.gettempdir(), "f1_zoo_panel.pkl")
        panel.to_pickle(PANEL_PATH)
        PANEL = PANEL_PATH
    wk = res.weekends
    allt = wk[[(wk["t"] < t).sum() >= 8 for t in wk["t"]]]
    ts = {"dev": sorted(allt[allt.year == 2024]["t"]), "val": sorted(allt[allt.year == 2025]["t"])}
    bonus = zoo.team_bonus(truth, ctruth)
    # references (continuous walk-forward pass, as in phases 4-6)
    refs_t = allt[allt["year"] <= 2025].reset_index(drop=True)
    base = ens.base_forecasts(res, truth, ctruth, refs_t)
    wide = ens.wide_table(truth, base)
    nn, _ = ens.rolling_nnls(wide, ["sim", "gbm"], refs_t["t"])
    refs = {"ref_sim_default": base["sim"], "ref_ewma_0.2": zoo.with_constructors(base["ewma"], truth, bonus),
            "ref_nnls(sim,gbm)": ens.adjust_constructors(base["sim"], nn, wide)}
    mlflow = X._mlflow()
    import mlflow.sklearn
    jobs = [(f, s, sp, ts[sp]) for f in FAMILIES for s in FSETS for sp in ("dev", "val")]
    with mlflow.start_run(run_name=f"phase7a-zoo {datetime.date.today()}"):
        mlflow.set_tags({"phase": os.environ.get("PHASE_TAG", "7a-zoo"), "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                         "families": ",".join(FAMILIES), "feature_sets": ",".join(FSETS), "primary_metric": "rmse"})
        rows, ref_pw = [], {}
        for split in ("dev", "val"):
            for name, fc in refs.items():
                sub = ens.subset(fc, set(ts[split]))
                pw = zoo.metrics(sub, truth, ctruth)
                ref_pw[(name, split)] = pw
                s = zoo.summary(pw)
                rows.append({"model": name, "family": name, "fset": "-", "split": split, **s})
                with mlflow.start_run(run_name=f"{name} [{split}]", nested=True):
                    mlflow.set_tags({"phase": "7a", "split": split, "data_version": X.DATA_VERSION, "kind": "reference"})
                    mlflow.log_metrics(s)
        with ProcessPoolExecutor(max_workers=6) as ex:
            for family, fset, split, fc, secs, blob in ex.map(_job, jobs):
                fc = zoo.with_constructors(fc, truth, bonus)
                pw = zoo.metrics(fc, truth, ctruth)
                s = zoo.summary(pw)
                bt = X.paired_bootstrap(pw, ref_pw[("ref_sim_default", split)], "rmse")
                bn = X.paired_bootstrap(pw, ref_pw[("ref_nnls(sim,gbm)", split)], "rmse")
                rows.append({"model": f"{family}/{fset}", "family": family, "fset": fset, "split": split, **s,
                             "vs_sim_rmse": bt["delta"], "vs_prevbest_rmse": bn["delta"], "seconds": secs})
                with mlflow.start_run(run_name=f"{family} / {fset} [{split}]", nested=True):
                    mlflow.set_tags({"phase": "7a", "split": split, "eval_year": X.SPLITS[split], "family": family,
                                     "feature_set": fset, "data_version": X.DATA_VERSION, "git_sha": X.git_sha()})
                    mlflow.log_params({"family": family, "feature_set": fset, "n_features": len(zoo.FEATURES[fset]),
                                       "hyperparams": "defaults", "refit_every": 1})
                    mlflow.log_metrics({**s, "seconds": secs, **{f"vs_sim_rmse_{k}": v for k, v in bt.items() if k != "n"},
                                        **{f"vs_prevbest_rmse_{k}": v for k, v in bn.items() if k != "n"}})
                    for r in pw.sort_values("t").itertuples():
                        mlflow.log_metrics({"wk_rmse": r.rmse, "wk_mae": r.mae}, step=int(r.round))
                    with tempfile.TemporaryDirectory() as td:
                        pw.to_csv(f"{td}/per_weekend.csv", index=False)
                        fc.drivers.to_csv(f"{td}/driver_predictions.csv", index=False)
                        mlflow.log_artifacts(td)
                    mlflow.sklearn.log_model(pickle.loads(blob), name="model", serialization_format="cloudpickle")
                print(f"{split} {family:10s} {fset:10s} rmse {s['rmse']:.3f} mae {s['mae']:.3f} rho {s['spearman']:.3f} pick5 {s['pick5_ratio']:.3f} c_mae {s.get('c_mae', np.nan):.2f}  [{secs:.0f}s]", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv("results/zoo_phase7a.csv", index=False)
        mlflow.log_artifact("results/zoo_phase7a.csv")
    piv = df.pivot_table(index="model", columns="split", values=["rmse", "mae", "spearman", "pick5_ratio"])
    print("\n", piv.sort_values(("rmse", "dev")).round(3).to_string())
