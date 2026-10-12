"""Run the whole study in order on the current data (about 15 minutes). Each phase logs to MLflow."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["phase1_baselines", "phase2_gbm_blend", "phase3_sim_ablation", "phase3b_random_search", "phase4_stacking",
         "phase5a_constructors", "phase6_practice", "phase6b_residual", "phase5b_holdout_2026", "phase5c_exploratory_2026"]
for name in ORDER:
    t0 = time.time()
    log = os.path.join(os.environ.get("LOGDIR", HERE), f"{name}.log")
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, os.path.join(HERE, f"{name}.py")], stdout=f, stderr=subprocess.STDOUT,
                             cwd=os.path.dirname(HERE), env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONPATH": os.path.dirname(HERE)})
    print(f"{name:28s} exit {rc}  {time.time() - t0:5.0f}s  -> {log}", flush=True)
    if rc != 0:
        sys.exit(f"{name} failed; see {log}")
print("ALL DONE", flush=True)
