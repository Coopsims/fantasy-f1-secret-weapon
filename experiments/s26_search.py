"""Nested 2026 search: Optuna per family on gamedays 1-9 (objective: driver RMSE vs official points), best spec of each
family scored on gamedays 10-17 (not used for tuning), plus pre-declared ensembles.  Everything logged to MLflow."""
import os, sys, datetime, json, pickle, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

CACHE = os.environ.get("S26_CACHE", "/tmp/s26_cache.pkl")
PANEL = os.environ["PANEL_PKL"]
N_TRIALS = int(os.environ.get("N_TRIALS", "150"))
TUNE = (int(os.environ.get("TUNE_LO", "1")), int(os.environ.get("TUNE_HI", "9")))
TEST = (int(os.environ.get("TEST_LO", "10")), int(os.environ.get("TEST_HI", "17")))
TAG = os.environ.get("RUN_TAG", "nested")
BEST_FROM_LOG = os.environ.get("BEST_FROM_LOG")          # re-use specs printed by an earlier run (skip tuning)


def _safe(name):
    import re
    return re.sub(r"[^A-Za-z0-9_.\- ]", "_", name)


def _specs_from_log(path):
    import re, ast
    best = {}
    for line in open(path):
        m = re.match(r"^(\w+): best tune RMSE [\d.]+ \| (\{.*\}) \| hp (\{.*\})$", line.strip())
        if m:
            spec = json.loads(m.group(2)); spec["hp"] = ast.literal_eval(m.group(3)); best[m.group(1)] = spec
    return best
TUNE_PAR = os.environ.get("TUNE_PARITY") or None          # 'odd' -> tune on odd gamedays, test on even (interleaved)
TEST_PAR = {"odd": "even", "even": "odd"}.get(TUNE_PAR)


def _study(args):
    family, n_trials, seed = args
    import optuna
    from experiments import s26
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    D = s26.prepare(PANEL, CACHE)
    tts = s26.gamedays(D, *TUNE, TUNE_PAR)
    recs = []

    def objective(trial):
        spec = s26.suggest_spec(trial, family)
        t0 = time.time()
        pw = s26.score(D, s26.predict_spec(D, spec, tts))
        s = s26.summary(pw)
        recs.append({"trial": trial.number, "spec": spec, "params": dict(trial.params), "seconds": time.time() - t0, **s})
        return s["rmse"]

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    warm = os.environ.get("WARM_FROM_LOG")
    if warm:                                                    # start from an earlier run's best spec for this family
        sp = _specs_from_log(warm).get(family)
        if sp:
            raw = sp["c_ewma"] / max(1e-9, 1 - max(sp["b_early"], sp["b_late"]))
            st.enqueue_trial({**sp["hp"], "fset": sp["fset"], "half_life": sp["half_life"], "w26": sp["w26"],
                              "use_official": sp["use_official"], "ewma_alpha": sp["ewma_alpha"], "b_early": sp["b_early"],
                              "b_late": sp["b_late"], "kappa": sp["kappa"], "c_ewma_raw": min(0.5, raw)})
    st.optimize(objective, n_trials=n_trials)
    best = min(recs, key=lambda r: r["rmse"])
    return family, best["spec"], recs


def _eval(args):
    name, spec, ts = args
    from experiments import s26
    D = s26.prepare(PANEL, CACHE)
    pred, models = s26.predict_spec(D, spec, ts, return_models=True)
    return name, pred, pickle.dumps(models[-1])


if __name__ == "__main__":
    import mlflow, mlflow.sklearn
    import mlflow_experiments as X
    from experiments import s26
    D = s26.prepare(PANEL, CACHE)
    tts, ets = s26.gamedays(D, *TUNE, TUNE_PAR), s26.gamedays(D, *TEST, TEST_PAR)
    print(f"tune on gamedays {TUNE} ({len(tts)} weekends), score on {TEST} ({len(ets)} weekends)", flush=True)
    # references
    P = D["panel"]
    def ref_pred(col, ts):
        return P[P["t"].isin(ts)][["t", "driver", col]].rename(columns={col: "pred"}).dropna()
    b = pd.read_csv("results/zoo_phase7b.csv")
    import re
    log7b = open(os.environ["PHASE7B_LOG"]).read()
    hp7 = {}
    for f in ["xgb", "catboost", "extratrees", "rf"]:
        m = re.search(rf"^{f}/(\w+): \d+ trials, .*best params (\{{.*\}})$", log7b, re.M)
        hp7[f] = eval(m.group(2))
    plain = lambda f: {"family": f, "hp": hp7[f], "fset": "all", "half_life": 999, "w26": 1.0, "use_official": False,
                       "ewma_alpha": 0.3, "b_sim": 0.0, "c_ewma": 0.0}
    refs = {"ref_sim(dotd)": lambda ts: ref_pred("sim", ts), "ref_ewma_off_0.3": lambda ts: ref_pred("ew_off_0.3", ts)}
    mlflow_ = X._mlflow()
    with mlflow_.start_run(run_name=f"s26-search {TAG} {datetime.date.today()}"):
        mlflow_.set_tags({"phase": "s26-search", "tag": TAG, "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                          "truth": "official 2026 fantasy points, own-car weekends", "tune_gamedays": str(TUNE),
                          "test_gamedays": str(TEST), "n_trials": N_TRIALS,
                          "tune_parity": str(TUNE_PAR), "test_parity": str(TEST_PAR)})
        with ProcessPoolExecutor(max_workers=5) as ex:
            ref7 = dict((n, p) for n, p, _ in ex.map(_eval, [(f"ref_phase7_{f}", plain(f), tts + ets) for f in hp7]))
        ref7["ref_phase7_mean4"] = pd.concat(ref7.values()).groupby(["t", "driver"], as_index=False)["pred"].mean()
        m4 = ref7["ref_phase7_mean4"].merge(P[["t", "driver", "sim"]], on=["t", "driver"])
        ref7["ref_phase7_mean4+sim"] = m4.assign(pred=0.5 * m4["pred"] + 0.5 * m4["sim"].fillna(m4["pred"]))[["t", "driver", "pred"]]
        allrefs = {**{k: f(tts + ets) for k, f in refs.items()}, **{k: v[["t", "driver", "pred"]] for k, v in ref7.items()}}
        rows, pw_test = [], {}
        for name, pred in allrefs.items():
            for split, ts in (("tune", tts), ("test", ets)):
                pw = s26.score(D, pred[pred["t"].isin(ts)])
                if split == "test":
                    pw_test[name] = pw
                s = s26.summary(pw)
                rows.append({"name": name, "split": split, **s})
                with mlflow_.start_run(run_name=f"{name} [{split}]", nested=True):
                    mlflow_.set_tags({"phase": "s26", "kind": "reference", "split": split, "tag": TAG})
                    mlflow_.log_metrics(s)
        print(pd.DataFrame(rows).pivot_table(index="name", columns="split", values=["rmse", "mae", "pick5_ratio"]).round(3).to_string(), flush=True)
        # ---- tuning ----
        t0 = time.time()
        if BEST_FROM_LOG:
            studies = [(f, sp, []) for f, sp in _specs_from_log(BEST_FROM_LOG).items()]
            print(f"specs re-used from {BEST_FROM_LOG}: {[f for f, _, _ in studies]} (their trials are logged in the earlier run)", flush=True)
            mlflow_.set_tag("specs_from", BEST_FROM_LOG)
        else:
            with ProcessPoolExecutor(max_workers=len(s26.FAMILIES)) as ex:
                studies = list(ex.map(_study, [(f, N_TRIALS, 11 + i) for i, f in enumerate(s26.FAMILIES)]))
        print(f"tuning done in {(time.time() - t0) / 60:.0f} min", flush=True)
        best = {}
        for family, spec, recs in studies:
            best[family] = spec
            for r in recs:
                with mlflow_.start_run(run_name=f"{family} trial {r['trial']:03d}", nested=True):
                    mlflow_.set_tags({"phase": "s26-trial", "family": family, "split": "tune", "tag": TAG})
                    mlflow_.log_params({k: (str(v)[:240] if isinstance(v, (str, bool)) else v) for k, v in r["params"].items()} | {"family": family})
                    mlflow_.log_metrics({k: r[k] for k in ("rmse", "mae", "spearman", "pick5_ratio", "bias", "seconds")})
            if recs:
                r0 = min(recs, key=lambda r: r["rmse"])
                print(f"{family}: best tune RMSE {r0['rmse']:.3f} | {json.dumps({k: v for k, v in spec.items() if k != 'hp'})} | hp {spec['hp']}", flush=True)
        # ---- score each family's best on the held-out gamedays ----
        with ProcessPoolExecutor(max_workers=len(best)) as ex:
            evals = list(ex.map(_eval, [(f, best[f], tts + ets) for f in best]))
        preds = {}
        for family, pred, blob in evals:
            preds[family] = pred
            for split, ts in (("tune", tts), ("test", ets)):
                pw = s26.score(D, pred[pred["t"].isin(ts)])
                s = s26.summary(pw)
                extra = {}
                if split == "test":
                    pw_test[family] = pw
                    for ref in ("ref_sim(dotd)", "ref_phase7_mean4+sim"):
                        bt = X.paired_bootstrap(pw, pw_test[ref], "rmse")
                        extra |= {f"vs_{_safe(ref)}_{k}": v for k, v in bt.items() if k != "n"}
                rows.append({"name": f"tuned_{family}", "split": split, **s, **extra})
                with mlflow_.start_run(run_name=f"tuned {family} [{split}]", nested=True):
                    mlflow_.set_tags({"phase": "s26-best", "family": family, "split": split, "tag": TAG})
                    mlflow_.log_params({"spec": json.dumps(best[family], default=str)[:4900]})
                    mlflow_.log_metrics(s | extra)
                    if split == "test":
                        with tempfile.TemporaryDirectory() as td:
                            pw.to_csv(f"{td}/per_weekend.csv", index=False); pred.to_csv(f"{td}/driver_predictions.csv", index=False)
                            mlflow_.log_artifacts(td)
                        mlflow.sklearn.log_model(pickle.loads(blob), name="model", serialization_format="cloudpickle")
        # pre-declared ensembles: mean of all family bests; mean of the 3 best by TUNE rmse
        tune_rank = sorted(best, key=lambda f: next(r["rmse"] for r in rows if r["name"] == f"tuned_{f}" and r["split"] == "tune"))
        for ename, members in (("ens_mean_all", list(best)), ("ens_mean_top3", tune_rank[:3])):
            pred = pd.concat([preds[f] for f in members]).groupby(["t", "driver"], as_index=False)["pred"].mean()
            preds[ename] = pred
            for split, ts in (("tune", tts), ("test", ets)):
                pw = s26.score(D, pred[pred["t"].isin(ts)])
                s = s26.summary(pw)
                extra = {}
                if split == "test":
                    pw_test[ename] = pw
                    for ref in ("ref_sim(dotd)", "ref_phase7_mean4+sim"):
                        bt = X.paired_bootstrap(pw, pw_test[ref], "rmse")
                        extra |= {f"vs_{_safe(ref)}_{k}": v for k, v in bt.items() if k != "n"}
                rows.append({"name": ename, "split": split, **s, **extra, "members": ",".join(members)})
                with mlflow_.start_run(run_name=f"{ename} [{split}]", nested=True):
                    mlflow_.set_tags({"phase": "s26-ens", "members": ",".join(members), "split": split, "tag": TAG})
                    mlflow_.log_metrics(s | extra)
        df = pd.DataFrame(rows)
        out = f"results/s26_{TAG}.csv"
        df.to_csv(out, index=False)
        pd.to_pickle({"best": best, "preds": preds}, f"results/s26_{TAG}_best.pkl")
        mlflow_.log_artifact(out)
    show = df[df.split == "test"].sort_values("rmse")
    cols = [c for c in ["name", "rmse", "mae", "spearman", "pick5_ratio", "bias", "vs_ref_sim_dotd__delta", "vs_ref_sim_dotd__lo",
                        "vs_ref_sim_dotd__hi", "vs_ref_phase7_mean4_sim_delta", "vs_ref_phase7_mean4_sim_lo", "vs_ref_phase7_mean4_sim_hi"] if c in show]
    print("\nheld-out gamedays:\n", show[cols].round(3).to_string(index=False))
