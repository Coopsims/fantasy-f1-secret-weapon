"""Phase 7b: Optuna tuning of the top model families from phase 7a (pre-declared: top 5 by dev RMSE, each with its best
feature set).  Objective = dev (2024) RMSE, walk-forward with a refit every 3 weekends (causal, for speed).  Every trial
is an MLflow run; the best setting per family is re-scored on dev and val with a refit every weekend, logged with its
fitted model, and compared with its default setting and the references (paired bootstrap)."""
import os, sys, datetime, pickle, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

N_TRIALS = {"ridge": 40, "elasticnet": 60, "huber": 60, "knn": 60}
DEFAULT_TRIALS = int(os.environ.get("N_TRIALS", "100"))
PANEL = os.environ["PANEL_PKL"]


def _data():
    import mlflow_experiments as X
    res, truth, ctruth = X.get_data()
    return res, truth, ctruth, pd.read_pickle(PANEL)


def _study(args):
    family, fset, dev_ts, n_trials = args
    import optuna
    from experiments import zoo
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    res, truth, ctruth, panel = _data()
    feats = zoo.FEATURES[fset]
    records = []

    def objective(trial):
        params = zoo.suggest(trial, family)
        t0 = time.time()
        fc = zoo.walk_forward(panel, feats, family, params, dev_ts, refit_every=3)
        pw = zoo.metrics(fc, truth, ctruth)
        s = zoo.summary(pw)
        records.append({"trial": trial.number, "params": params, "seconds": time.time() - t0, **s})
        return s["rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    study.enqueue_trial({k: v for k, v in _defaults(family).items() if k in zoo.SPACES[family]})   # trial 0 = defaults
    study.optimize(objective, n_trials=n_trials)
    return family, fset, study.best_params, records


def _defaults(family):
    import inspect
    from experiments import zoo
    src = inspect.getsource(zoo.make_model)
    out = {}
    for k, spec in zoo.SPACES[family].items():           # default values written as p.get("name", default)
        import re
        m = re.search(rf'p\.get\("{k}", ([^)]+)\)', src[src.index(f'"{family}"'):])
        if m:
            v = eval(m.group(1).split(")")[0])
            out[k] = v if spec[0] != "int" else int(v)
    return out


def _final(args):
    family, fset, params, ts_dev, ts_val = args
    from experiments import zoo
    res, truth, ctruth, panel = _data()
    feats = zoo.FEATURES[fset]
    out = {}
    for split, ts in (("dev", ts_dev), ("val", ts_val)):
        for tag, p in (("tuned", params), ("default", None)):
            out[(split, tag)] = zoo.walk_forward(panel, feats, family, p, ts)
    tr = panel[(panel["t"] < max(ts_val)) & panel["d_last1"].notna()]
    model = zoo.make_model(family, params).fit(tr[feats], tr["total"])
    return family, fset, params, out, pickle.dumps(model)


if __name__ == "__main__":
    import mlflow_experiments as X
    from experiments import zoo, ens
    import mlflow.sklearn
    res, truth, ctruth, panel = _data()
    wk = res.weekends
    allt = wk[[(wk["t"] < t).sum() >= 8 for t in wk["t"]]]
    ts_dev, ts_val = sorted(allt[allt.year == 2024]["t"]), sorted(allt[allt.year == 2025]["t"])
    a = pd.read_csv("results/zoo_phase7a.csv")
    dev = a[(a.split == "dev") & (a.fset != "-")]
    best_fset = dev.loc[dev.groupby("family")["rmse"].idxmin()].sort_values("rmse")
    chosen = best_fset.head(int(os.environ.get("TOP", "5")))[["family", "fset", "rmse"]].values.tolist()
    print("tuning (pre-declared top families by dev RMSE):", chosen, flush=True)
    bonus = zoo.team_bonus(truth, ctruth)
    mlflow = X._mlflow()
    with mlflow.start_run(run_name=f"phase7b-optuna {datetime.date.today()}"):
        mlflow.set_tags({"phase": "7b-optuna", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                         "objective": "dev 2024 RMSE, walk-forward refit every 3", "chosen": str(chosen)})
        jobs = [(f, s, ts_dev, N_TRIALS.get(f, DEFAULT_TRIALS)) for f, s, _ in chosen]
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            studies = list(ex.map(_study, jobs))
        for family, fset, best, records in studies:
            for r in records:
                with mlflow.start_run(run_name=f"{family}/{fset} trial {r['trial']:03d}", nested=True):
                    mlflow.set_tags({"phase": "7b-trial", "family": family, "feature_set": fset, "split": "dev",
                                     "data_version": X.DATA_VERSION})
                    mlflow.log_params({**{f"hp.{k}": v for k, v in r["params"].items()}, "family": family, "feature_set": fset,
                                       "trial": r["trial"], "refit_every": 3})
                    mlflow.log_metrics({k: r[k] for k in ("rmse", "mae", "spearman", "top5", "pick5_ratio", "seconds") if k in r})
            print(f"{family}/{fset}: {len(records)} trials, best dev rmse {min(r['rmse'] for r in records):.3f} "
                  f"(trial 0 = defaults: {records[0]['rmse']:.3f}); best params {best}", flush=True)
        refs = {}
        refs_t = allt[allt["year"] <= 2025].reset_index(drop=True)
        base = ens.base_forecasts(res, truth, ctruth, refs_t)
        wide = ens.wide_table(truth, base)
        nn, _ = ens.rolling_nnls(wide, ["sim", "gbm"], refs_t["t"])
        for sp, ts in (("dev", ts_dev), ("val", ts_val)):
            refs[("sim", sp)] = zoo.metrics(ens.subset(base["sim"], set(ts)), truth, ctruth)
            refs[("prev", sp)] = zoo.metrics(ens.subset(ens.adjust_constructors(base["sim"], nn, wide), set(ts)), truth, ctruth)
        rows = []
        with ProcessPoolExecutor(max_workers=len(studies)) as ex:
            finals = list(ex.map(_final, [(f, s, b, ts_dev, ts_val) for f, s, b, _ in studies]))
        preds = {}
        for family, fset, params, out, blob in finals:
            for (split, tag), fc in out.items():
                fc = zoo.with_constructors(fc, truth, bonus)
                preds[(family, fset, split, tag)] = fc
                pw = zoo.metrics(fc, truth, ctruth)
                s = zoo.summary(pw)
                vs_sim = X.paired_bootstrap(pw, refs[("sim", split)], "rmse")
                vs_prev = X.paired_bootstrap(pw, refs[("prev", split)], "rmse")
                rows.append({"family": family, "fset": fset, "split": split, "setting": tag, **s,
                             **{f"vs_sim_{k}": v for k, v in vs_sim.items()}, **{f"vs_prev_{k}": v for k, v in vs_prev.items()}})
                with mlflow.start_run(run_name=f"{family}/{fset} {tag} [{split}]", nested=True):
                    mlflow.set_tags({"phase": "7b-final", "family": family, "feature_set": fset, "split": split,
                                     "setting": tag, "data_version": X.DATA_VERSION, "git_sha": X.git_sha()})
                    mlflow.log_params({**{f"hp.{k}": v for k, v in (params if tag == "tuned" else {}).items()},
                                       "family": family, "feature_set": fset, "setting": tag, "refit_every": 1})
                    mlflow.log_metrics({**s, **{f"vs_sim_rmse_{k}": v for k, v in vs_sim.items() if k != "n"},
                                        **{f"vs_prev_rmse_{k}": v for k, v in vs_prev.items() if k != "n"}})
                    with tempfile.TemporaryDirectory() as td:
                        pw.to_csv(f"{td}/per_weekend.csv", index=False)
                        fc.drivers.to_csv(f"{td}/driver_predictions.csv", index=False)
                        mlflow.log_artifacts(td)
                    if tag == "tuned" and split == "val":
                        mlflow.sklearn.log_model(pickle.loads(blob), name="model", serialization_format="cloudpickle")
        df = pd.DataFrame(rows)
        df.to_csv("results/zoo_phase7b.csv", index=False)
        pd.to_pickle(preds, os.path.join(tempfile.gettempdir(), "f1_zoo_7b_preds.pkl"))
        mlflow.log_artifact("results/zoo_phase7b.csv")
    cols = ["family", "fset", "split", "setting", "rmse", "mae", "spearman", "pick5_ratio", "c_mae", "vs_sim_delta", "vs_sim_lo",
            "vs_sim_hi", "vs_prev_delta", "vs_prev_lo", "vs_prev_hi"]
    print(df[cols].sort_values(["split", "rmse"]).round(3).to_string(index=False))
