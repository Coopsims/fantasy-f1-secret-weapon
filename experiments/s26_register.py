"""Fit the selected 2026 spec ensemble on every completed weekend and register it in the MLflow Model Registry."""
import os, sys, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

if __name__ == "__main__":
    import mlflow, mlflow.pyfunc
    import mlflow_experiments as X
    import f1_planner as FP
    from experiments import s26
    spec = json.load(open(sys.argv[1] if len(sys.argv) > 1 else FP.SPEC_FILE))
    D = s26.prepare(os.environ["PANEL_PKL"], os.environ["S26_CACHE"])
    P = D["panel"]
    t_now = int(P["t"].max()) + 1
    models, feats = [], []
    members = FP.expand_seeds(spec["specs"])
    for sp in members:
        m, f = FP.fit_tree(sp, P, t_now)
        models.append(m); feats.append(f)
    model = FP.FantasyPointsModel(models, feats, members)
    sample = P[P["t"] == P["t"].max()].head(5)
    cols = sorted(set(sum(feats, [])) | {"sim"} | {f"ew_off_{sp['ewma_alpha']}" for sp in members})
    alias = os.environ.get("ALIAS", "champion")
    mlflow_ = X._mlflow()
    with mlflow_.start_run(run_name=f"register {spec['name'][:80]}"):
        mlflow_.set_tags({"phase": "s26-register", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(),
                          "evidence": spec.get("evidence", "")[:4900], "fitted_on": f"all weekends with t < {t_now} (through the latest race)",
                          "constructors": f"derived: sum of drivers + official bonus EWMA alpha {spec.get('bonus_alpha')} (c_sim {spec.get('c_sim')})"})
        mlflow_.log_params({"n_specs": len(spec["specs"]), "spec_json": json.dumps(spec["specs"], default=float)[:5900]})
        from mlflow.models import infer_signature
        sig = infer_signature(sample[cols], model.predict(None, sample[cols], {"n26": 17}), params={"n26": 17})
        mlflow.pyfunc.log_model(name="model", python_model=model, input_example=sample[cols], signature=sig,
                                registered_model_name="f1-fantasy-points-2026", code_paths=["f1_planner.py"])
        mlflow_.log_artifact(sys.argv[1] if len(sys.argv) > 1 else FP.SPEC_FILE)
    client = mlflow.tracking.MlflowClient()
    ver = max(int(v.version) for v in client.search_model_versions("name='f1-fantasy-points-2026'"))
    client.set_registered_model_alias("f1-fantasy-points-2026", alias, str(ver))
    loaded = mlflow.pyfunc.load_model(f"models:/f1-fantasy-points-2026@{alias}")
    chk = loaded.predict(sample[cols], params={"n26": 17})
    print(f"registered f1-fantasy-points-2026 v{ver} alias '{alias}'; reload check: {np.round(chk, 1)} vs direct {np.round(model.predict(None, sample[cols], {'n26': 17}), 1)}")
