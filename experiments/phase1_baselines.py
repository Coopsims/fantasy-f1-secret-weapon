"""Phase 1: log the reference forecasters (naive baselines, GBM, default simulator) on dev (2024) and val (2025)."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlflow_experiments as X

mlflow = X._mlflow()
models = [
    ("sim_default", X.Sim()),
    ("baseline_last1", X.Baseline("last1")), ("baseline_last3", X.Baseline("last3")),
    ("baseline_last5", X.Baseline("last5")), ("baseline_season_mean", X.Baseline("season")),
    ("baseline_ewma_0.2", X.Baseline("ewma", 0.2)), ("baseline_ewma_0.3", X.Baseline("ewma", 0.3)),
    ("baseline_ewma_0.5", X.Baseline("ewma", 0.5)),
    ("gbm_default", X.GBM()), ("gbm_mae_objective", X.GBM(objective="reg:absoluteerror")),
]
with mlflow.start_run(run_name=f"phase1-baselines {datetime.date.today()}"):
    mlflow.set_tags({"phase": "1-baselines", "git_sha": X.git_sha()})
    for split in ("dev", "val"):
        ref = None
        for name, m in models:
            fc, pw, s = X.run_and_log(f"{name} [{split}]", m, split, ref=ref, tags={"phase": "1"})
            if name == "sim_default":
                ref = ("sim_default", pw)
            print(f"{split} {name:22s} MAE {s['mae']:.3f}  rho {s['spearman']:.3f}  top5 {s['top5']:.3f}")
