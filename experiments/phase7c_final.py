"""Phase 7c: ensembles of the tuned models, finalist chosen on val (2025), ONE pre-registered test on 2026, and the
finalist registered in the MLflow Model Registry.

Candidate list and rule, fixed before any ensemble was scored:
  singles   xgb, catboost, extratrees, rf (tuned in phase 7b)
  C_mean    equal-weight mean of the four
  C_nnls4   adaptive non-negative weights of the four (re-fit on all earlier predicted weekends)
  C_nnls5   the same plus the simulator
  C_mix     adaptive weights of (mean of the four, simulator)
  rule      finalist = lowest val RMSE; the runner-up is also scored on 2026; references: simulator, previous best
            ensemble nnls(simulator, GBM).  Adopt the finalist iff its 2026 RMSE change vs the simulator has a 95% CI
            excluding 0 (same rule as phase 5).
"""
import os, sys, datetime, json, pickle, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

PANEL = os.environ["PANEL_PKL"]
TREES = ["xgb", "catboost", "extratrees", "rf"]


def _wf(args):
    family, fset, params, ts = args
    from experiments import zoo
    panel = pd.read_pickle(PANEL)
    return family, zoo.walk_forward(panel, zoo.FEATURES[fset], family, params, ts)


def _fit_full(args):
    family, fset, params, last_t = args
    from experiments import zoo
    panel = pd.read_pickle(PANEL)
    tr = panel[(panel["t"] <= last_t) & panel["d_last1"].notna()]
    return family, pickle.dumps(zoo.make_model(family, params).fit(tr[zoo.FEATURES[fset]], tr["total"]))


def combine(fcs, names, how, truth, ts):
    """how = 'mean' or 'nnls' over the named forecasts (drivers only)."""
    from experiments import ens
    import mlflow_experiments as X
    w = truth[["t", "driver", "team", "total"]].copy()
    for n in names:
        w = w.merge(fcs[n].drivers[["t", "driver", "pred"]].rename(columns={"pred": n}), on=["t", "driver"], how="inner")
    if how == "mean":
        out = w[w["t"].isin(ts)].assign(pred=lambda d: d[names].mean(1))[["t", "driver", "pred"]]
        return X.Forecast(out.reset_index(drop=True)), None
    fc, wts = ens.rolling_nnls(w, names, sorted(ts), min_rows=200)
    return fc, wts


if __name__ == "__main__":
    import mlflow, mlflow.pyfunc, mlflow.sklearn
    import mlflow_experiments as X
    from experiments import zoo, ens
    res, truth, ctruth = X.get_data()
    panel = pd.read_pickle(PANEL)
    b = pd.read_csv("results/zoo_phase7b.csv")
    import re
    log = open(os.environ["PHASE7B_LOG"]).read()
    params = {}
    for f in TREES:                                            # tuned settings printed by phase 7b
        m = re.search(rf"^{f}/(\w+): \d+ trials, .*best params (\{{.*\}})$", log, re.M)
        params[f] = (m.group(1), eval(m.group(2)))
    wk = res.weekends
    allt = wk[[(wk["t"] < t).sum() >= 8 for t in wk["t"]]]
    ts = {y: sorted(allt[allt.year == y]["t"]) for y in (2024, 2025, 2026)}
    bonus = zoo.team_bonus(truth, ctruth)
    # walk-forward predictions of every tuned tree model for 2024-2026 (2026 is NOT looked at until the pre-registration)
    jobs = [(f, params[f][0], params[f][1], ts[2024] + ts[2025] + ts[2026]) for f in TREES]
    with ProcessPoolExecutor(max_workers=4) as ex:
        fcs = dict(ex.map(_wf, jobs))
    base = ens.base_forecasts(res, truth, ctruth, allt.reset_index(drop=True))
    fcs["sim"] = base["sim"]
    wide = ens.wide_table(truth, base)
    prev, _ = ens.rolling_nnls(wide, ["sim", "gbm"], sorted(allt["t"]))
    mean4, _ = combine(fcs, TREES, "mean", truth, ts[2024] + ts[2025] + ts[2026])
    fcs["mean4"] = mean4
    cands = {f: fcs[f] for f in TREES}
    cands["C_mean"] = mean4
    cands["C_nnls4"], w4 = combine(fcs, TREES, "nnls", truth, ts[2025] + ts[2026])
    cands["C_nnls5"], w5 = combine(fcs, TREES + ["sim"], "nnls", truth, ts[2025] + ts[2026])
    cands["C_mix"], wm = combine(fcs, ["mean4", "sim"], "nnls", truth, ts[2025] + ts[2026])
    refs = {"ref_sim": base["sim"], "ref_prev_best": ens.adjust_constructors(base["sim"], prev, wide)}

    def score(fc, year):
        sub = ens.subset(fc if fc.constructors is not None else zoo.with_constructors(fc, truth, bonus), set(ts[year]))
        return zoo.metrics(sub, truth, ctruth)

    mlflow_ = X._mlflow()
    rows = []
    with mlflow_.start_run(run_name=f"phase7c-ensembles-val {datetime.date.today()}"):
        mlflow_.set_tags({"phase": "7c-val", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "rule": __doc__.strip()[:480]})
        ref_val = {k: score(v, 2025) for k, v in refs.items()}
        for name, fc in {**cands, **refs}.items():
            pw = score(fc, 2025)
            s = zoo.summary(pw)
            vs = X.paired_bootstrap(pw, ref_val["ref_sim"], "rmse"); vp = X.paired_bootstrap(pw, ref_val["ref_prev_best"], "rmse")
            rows.append({"candidate": name, **s, "vs_sim": vs["delta"], "vs_sim_lo": vs["lo"], "vs_sim_hi": vs["hi"],
                         "vs_prev": vp["delta"], "vs_prev_lo": vp["lo"], "vs_prev_hi": vp["hi"]})
            with mlflow_.start_run(run_name=f"{name} [val]", nested=True):
                mlflow_.set_tags({"phase": "7c", "split": "val", "candidate": name, "data_version": X.DATA_VERSION})
                if name in params:
                    mlflow_.log_params({f"hp.{k}": v for k, v in params[name][1].items()} | {"feature_set": params[name][0]})
                mlflow_.log_metrics({**s, "vs_sim_rmse": vs["delta"], "vs_prev_rmse": vp["delta"]})
        val = pd.DataFrame(rows).sort_values("rmse")
        val.to_csv("results/zoo_phase7c_val.csv", index=False)
        mlflow_.log_artifact("results/zoo_phase7c_val.csv")
    print(val[["candidate", "rmse", "mae", "spearman", "pick5_ratio", "c_mae", "vs_sim", "vs_sim_lo", "vs_sim_hi", "vs_prev", "vs_prev_lo", "vs_prev_hi"]].round(3).to_string(index=False))
    pool = val[~val.candidate.str.startswith("ref_")]
    finalist, runner = pool.iloc[0]["candidate"], pool.iloc[1]["candidate"]
    print(f"\nfinalist (lowest val RMSE): {finalist} | runner-up: {runner}")

    # ---- pre-registration, then the single 2026 look ----
    with mlflow_.start_run(run_name="PRE-REGISTRATION 2026 holdout (phase 7)"):
        mlflow_.set_tags({"phase": "7-prereg", "finalist": finalist, "runner_up": runner, "references": "ref_sim, ref_prev_best",
                          "rule": "adopt iff 2026 RMSE change vs simulator has 95% CI excluding 0", "data_version": X.DATA_VERSION,
                          "registered_at": datetime.datetime.now().isoformat(timespec="seconds"), "git_sha": X.git_sha(),
                          "tuned_params": json.dumps({k: v[1] for k, v in params.items()})[:4900]})
    # official points as a second truth for 2026 (own-car weekends only)
    from f1_calendar import GRID_2026
    op = pd.read_csv("fantasy_f1_official_points.csv")
    t26 = truth[truth.year == 2026].merge(op[op.type == "driver"][["round", "name", "points"]], left_on=["round", "driver"], right_on=["round", "name"], how="inner")
    t26 = t26[t26["team"] == t26["driver"].map(GRID_2026)]
    truth_off = pd.concat([truth[truth.year != 2026], t26.assign(total=t26["points"])[truth.columns]], ignore_index=True)
    test_rows = []
    with mlflow_.start_run(run_name=f"phase7c-holdout-2026 {datetime.date.today()}"):
        mlflow_.set_tags({"phase": "7c-test", "data_version": X.DATA_VERSION, "git_sha": X.git_sha()})
        pw_ref = {k: score(v, 2026) for k, v in refs.items()}
        for name in [finalist, runner, "ref_sim", "ref_prev_best"]:
            fc = cands.get(name, refs.get(name))
            pw = score(fc, 2026)
            s = zoo.summary(pw)
            vs = X.paired_bootstrap(pw, pw_ref["ref_sim"], "rmse"); vp = X.paired_bootstrap(pw, pw_ref["ref_prev_best"], "rmse")
            sub = ens.subset(fc, set(ts[2026]))
            off = zoo.summary(zoo.metrics(X.Forecast(sub.drivers), truth_off, None))
            test_rows.append({"candidate": name, **s, "vs_sim": vs["delta"], "vs_sim_lo": vs["lo"], "vs_sim_hi": vs["hi"],
                              "vs_prev": vp["delta"], "vs_prev_lo": vp["lo"], "vs_prev_hi": vp["hi"],
                              "official_rmse": off["rmse"], "official_mae": off["mae"], "official_spearman": off["spearman"]})
            with mlflow_.start_run(run_name=f"{name} [test 2026]", nested=True):
                mlflow_.set_tags({"phase": "7c", "split": "test", "candidate": name, "data_version": X.DATA_VERSION})
                mlflow_.log_metrics({**s, "vs_sim_rmse": vs["delta"], "vs_sim_rmse_lo": vs["lo"], "vs_sim_rmse_hi": vs["hi"],
                                     "vs_prev_rmse": vp["delta"], "official_rmse": off["rmse"], "official_mae": off["mae"]})
        test = pd.DataFrame(test_rows)
        test.to_csv("results/zoo_phase7c_test.csv", index=False)
        mlflow_.log_artifact("results/zoo_phase7c_test.csv")
    print("\n2026 holdout (reconstructed points = training target; official points as a second truth):")
    print(test[["candidate", "rmse", "mae", "spearman", "pick5_ratio", "c_mae", "vs_sim", "vs_sim_lo", "vs_sim_hi", "vs_prev", "vs_prev_lo", "vs_prev_hi",
                "official_rmse", "official_mae"]].round(3).to_string(index=False))
    fin = test.iloc[0]
    adopted = fin["vs_sim_hi"] < 0
    print(f"\nrule: {finalist} {'ADOPTED' if adopted else 'NOT confirmed'} (2026 RMSE change vs simulator {fin['vs_sim']:+.3f} [{fin['vs_sim_lo']:+.3f}, {fin['vs_sim_hi']:+.3f}])")

    # ---- register the finalist (fitted on every weekend up to the latest) ----
    last_t = int(res.weekends["t"].max())
    comps = [finalist] if finalist in TREES else TREES
    with ProcessPoolExecutor(max_workers=4) as ex:
        fitted = dict(ex.map(_fit_full, [(f, params[f][0], params[f][1], last_t) for f in comps]))

    class Ensemble(mlflow.pyfunc.PythonModel):
        """Weighted mean of fitted tree models on the zoo feature panel (rows = driver-weekends)."""
        def __init__(self, models, feats, weights):
            self.models, self.feats, self.weights = models, feats, weights
        def predict(self, context, model_input, params=None):
            preds = np.column_stack([m.predict(model_input[self.feats[n]]) for n, m in self.models.items()])
            return preds @ np.array([self.weights[n] for n in self.models])

    models = {f: pickle.loads(b_) for f, b_ in fitted.items()}
    feats = {f: zoo.FEATURES[params[f][0]] for f in models}
    if finalist in TREES:
        weights = {finalist: 1.0}
    elif finalist == "C_mean" or finalist == "C_mix":
        weights = {f: 1 / len(TREES) for f in TREES}
    else:
        wt = (w4 if finalist == "C_nnls4" else w5).iloc[-1]
        weights = {f: float(wt[f]) for f in TREES}
    with mlflow_.start_run(run_name=f"register {finalist}"):
        mlflow_.set_tags({"phase": "7c-register", "finalist": finalist, "adopted_by_rule": str(adopted), "data_version": X.DATA_VERSION,
                          "fitted_through_t": str(last_t), "note": "C_mix / C_nnls5 also blend in the simulator, which is not part of this pyfunc"})
        mlflow_.log_params({"weights": json.dumps(weights), "feature_sets": json.dumps({f: params[f][0] for f in models})})
        sample = panel[panel["t"] == last_t].head(5)
        info = mlflow.pyfunc.log_model(name="model", python_model=Ensemble(models, feats, weights),
                                       input_example=sample[sorted(set(sum(feats.values(), [])))],
                                       registered_model_name="f1-driver-points")
        client = mlflow.tracking.MlflowClient()
        try:
            ver = max(int(m.version) for m in client.search_model_versions("name='f1-driver-points'"))
            client.set_registered_model_alias("f1-driver-points", "champion" if adopted else "candidate", str(ver))
            print(f"registered f1-driver-points v{ver} with alias '{'champion' if adopted else 'candidate'}'")
        except Exception as e:
            print("registry alias not set:", e)
