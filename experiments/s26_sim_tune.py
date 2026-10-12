"""Tune the weekend simulator for 2026: random search over ModelParams, objective = driver RMSE vs official points
(incl. Driver of the Day) on ODD 2026 gamedays; the best settings are reported on EVEN gamedays.  MLflow-logged."""
import os, sys, datetime, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

N = int(os.environ.get("N_CONFIGS", "120"))
CACHE = os.environ["S26_CACHE"]
PANEL = os.environ["PANEL_PKL"]
SPACE = {"half_life": (4, 40, True), "driver_ridge": (1, 12, True), "team_ridge_new": (0.3, 8, True),
         "team_ridge_old": (0.05, 1.5, True), "team_link": (0.2, 6, True), "link_rho": (0.3, 1.0, False),
         "sprint_weight": (0.2, 1.0, False), "weekend_shock": (0.1, 0.7, False), "chaos_sigma": (0.15, 0.6, False),
         "fl_beta": (0.2, 1.2, False), "prior_scale": (0.2, 1.6, False), "p0_max": (0.15, 0.30, False),
         "dnf_a_team": (5, 40, True), "dnf_a_driver": (2, 15, True), "dnf_half_life_mult": (0.5, 3.0, True)}


def _eval(args):
    i, cfg = args
    import mlflow_experiments as X
    from experiments import s26
    from f1_weekend_model import ModelParams
    res, truth, ctruth = X.get_data()
    D = s26.prepare(PANEL, CACHE)
    tg = res.weekends[res.weekends["t"].isin(D["ts26"])].reset_index(drop=True)
    fc = X.Sim(ModelParams(**cfg), dotd=True, n_sims=3000).predict(res, truth, ctruth, tg)
    pred = fc.drivers[["t", "driver", "pred"]]
    out = {}
    for par in ("odd", "even"):
        ts = s26.gamedays(D, 1, 17, par)
        out[par] = s26.summary(s26.score(D, pred[pred["t"].isin(ts)]))
    return i, cfg, out, pred, fc.constructors


if __name__ == "__main__":
    import mlflow_experiments as X
    rng = np.random.default_rng(2026)
    cfgs = [{}]
    for _ in range(N):
        cfgs.append({k: float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if lg else rng.uniform(lo, hi)) for k, (lo, hi, lg) in SPACE.items()})
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "6"))) as ex:
        outs = list(ex.map(_eval, list(enumerate(cfgs))))
    mlflow = X._mlflow()
    rows = []
    with mlflow.start_run(run_name=f"s26-sim-tune {datetime.date.today()}"):
        mlflow.set_tags({"phase": "s26-sim-tune", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "n_configs": N,
                         "objective": "RMSE vs official points on odd 2026 gamedays"})
        for i, cfg, out, _, _ in outs:
            rows.append({"cfg": i, **{f"odd_{k}": v for k, v in out["odd"].items()}, **{f"even_{k}": v for k, v in out["even"].items()}, **cfg})
            with mlflow.start_run(run_name=f"sim cfg {i:03d}", nested=True):
                mlflow.set_tags({"phase": "s26-sim-trial", "cfg": str(i)})
                mlflow.log_params(cfg or {"default": True})
                mlflow.log_metrics({f"odd_{k}": v for k, v in out["odd"].items()} | {f"even_{k}": v for k, v in out["even"].items()})
        df = pd.DataFrame(rows).sort_values("odd_rmse")
        df.to_csv("results/s26_sim_tune.csv", index=False)
        mlflow.log_artifact("results/s26_sim_tune.csv")
    best = df.iloc[0]
    print(df[["cfg", "odd_rmse", "even_rmse", "odd_mae", "even_mae", "even_pick5_ratio"]].head(8).round(3).to_string(index=False))
    print("default:", df[df.cfg == 0][["odd_rmse", "even_rmse"]].round(3).to_dict("records"))
    keep = {i: (pred, cons) for i, cfg, out, pred, cons in outs if i in (0, int(best.cfg))}
    pd.to_pickle({"best_cfg": cfgs[int(best.cfg)], "best_i": int(best.cfg), "preds": keep}, "results/cache/s26_sim_tune.pkl")
    print("best cfg:", json.dumps(cfgs[int(best.cfg)]))
