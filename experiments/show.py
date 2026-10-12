"""Print logged runs from MLflow as a table:  python experiments/show.py [tag=value ...]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
import mlflow_experiments as X
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 60)
mlflow = X._mlflow()
exp = mlflow.get_experiment_by_name(X.EXPERIMENT)
runs = mlflow.search_runs([exp.experiment_id], max_results=5000)
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    runs = runs[runs[f"tags.{k}"] == v]
cols = {"tags.mlflow.runName": "run", "tags.split": "split", "metrics.mae": "mae", "metrics.spearman": "rho", "metrics.top5": "top5",
        "metrics.cover80": "cov80", "metrics.c_mae": "c_mae", "metrics.vs_ref_mae_delta": "d_mae", "metrics.vs_ref_mae_lo": "lo",
        "metrics.vs_ref_mae_hi": "hi", "metrics.vs_ref_mae_p_a_lower": "P(better)", "metrics.vs_ref_spearman_delta": "d_rho"}
cols = {k: v for k, v in cols.items() if k in runs}
print(runs[list(cols)].rename(columns=cols).sort_values(["split", "run"]).round(3).to_string(index=False))
