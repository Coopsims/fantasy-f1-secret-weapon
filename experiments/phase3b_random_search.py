"""Phase 3b: random search over the simulator's continuous settings on dev (2024); top configs re-evaluated on val (2025).
The best-of-N dev score is optimistic by construction; val is the honest number."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mlflow_experiments as X
from f1_weekend_model import ModelParams
from experiments.parallel import eval_many

N = 96
SPACE = {  # (low, high, log?)
    "half_life": (4, 30, True), "driver_ridge": (1, 12, True), "team_ridge_new": (0.5, 8, True),
    "team_ridge_old": (0.05, 1.5, True), "team_link": (0.5, 6, True), "link_rho": (0.5, 1.0, False),
    "sprint_weight": (0.2, 1.0, False), "weekend_shock": (0.1, 0.7, False), "chaos_sigma": (0.15, 0.6, False),
    "fl_beta": (0.2, 1.2, False), "default_c": (1.2, 2.2, False), "prior_scale": (0.3, 1.6, False),
}

if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    cfgs = [{}]
    for _ in range(N):
        c = {}
        for k, (lo, hi, lg) in SPACE.items():
            c[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if lg else rng.uniform(lo, hi))
        cfgs.append(c)
    mlflow = X._mlflow()
    with mlflow.start_run(run_name=f"phase3b-random-search {datetime.date.today()}"):
        mlflow.set_tags({"phase": "3b-random-search", "git_sha": X.git_sha(), "n_configs": N, "seed": 2024})
        dev = eval_many(cfgs, "dev")
        ref = ("sim_default", dev[0][1])
        scores = []
        for i, (c, (fc, pw, _)) in enumerate(zip(cfgs, dev)):
            name = "default" if i == 0 else f"rs{i:03d}"
            _, _, s = X.run_and_log(f"{name} [dev]", X.Sim(ModelParams(**c)), "dev", ref=ref, tags={"phase": "3b", "cfg": name}, fc_override=fc)
            scores.append((s["mae"], i))
        scores.sort()
        print("default dev MAE", round(scores[[i for _, i in scores].index(0)][0], 3), "| best 5 dev:", [(round(a, 3), f"rs{i:03d}") for a, i in scores[:5]])
        top = [i for _, i in scores[:6] if i != 0][:5]
        val = eval_many([cfgs[0]] + [cfgs[i] for i in top], "val")
        vref = ("sim_default", val[0][1])
        print("val default MAE", round(X.summarize(val[0][1])["mae"], 3))
        for i, (fc, pw, _) in zip(top, val[1:]):
            _, _, s = X.run_and_log(f"rs{i:03d} [val]", X.Sim(ModelParams(**cfgs[i])), "val", ref=vref, tags={"phase": "3b", "cfg": f"rs{i:03d}"}, fc_override=fc)
            bt = X.paired_bootstrap(pw, val[0][1], "mae")
            print(f"rs{i:03d}: dev {dict((j, a) for a, j in scores)[i]:.3f} -> val {s['mae']:.3f} (delta vs default {bt['delta']:+.3f} CI [{bt['lo']:+.3f},{bt['hi']:+.3f}])  "
                  + " ".join(f"{k}={cfgs[i][k]:.2f}" for k in ("half_life", "driver_ridge", "team_ridge_new", "chaos_sigma", "fl_beta")))
