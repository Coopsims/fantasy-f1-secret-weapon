"""Stacking check: non-negative least-squares weights over the tuned families' walk-forward predictions, fitted on the
odd 2026 gamedays (official points), scored on the even ones.  Bar set before the run: replace the deployed ET+XGB
only if the stack's even RMSE is below 14.62 (ET+XGB's own, already optimistic, even score).  Logged to MLflow."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from scipy.optimize import nnls

if __name__ == "__main__":
    import mlflow_experiments as X
    from experiments import s26
    D = s26.prepare(os.environ["PANEL_PKL"], os.environ["S26_CACHE"])
    preds = {}
    for p in sys.argv[1:]:
        b = pd.read_pickle(p)
        preds |= {f: b["preds"][f] for f in b["best"]}
    fams = sorted(preds)
    w = None
    for f in fams:
        x = preds[f][["t", "driver", "pred"]].rename(columns={"pred": f})
        w = x if w is None else w.merge(x, on=["t", "driver"])
    w = w.merge(D["panel"][["t", "driver", "official"]], on=["t", "driver"]).dropna(subset=["official"])
    bad = w[fams].isna().any(axis=1)
    print("rows with a missing prediction:", int(bad.sum()), {f: int(w[f].isna().sum()) for f in fams if w[f].isna().any()})
    w = w[~bad]
    odd, even = s26.gamedays(D, 1, 17, "odd"), s26.gamedays(D, 1, 17, "even")
    tr = w[w["t"].isin(odd)]
    coef, _ = nnls(tr[fams].values, tr["official"].values)
    out = {}
    for name, c in (("nnls", coef), ("nnls_normalised", coef / coef.sum())):
        pr = w[["t", "driver"]].assign(pred=w[fams].values @ c)
        out[name] = {s: s26.summary(s26.score(D, pr[pr["t"].isin(ts)]))["rmse"] for s, ts in (("odd", odd), ("even", even))}
    print("weights:", dict(zip(fams, np.round(coef, 3))), "sum", round(coef.sum(), 3))
    print(out)
    verdict = "replace" if min(v["even"] for v in out.values()) < 14.62 else "keep ET+XGB"
    print("verdict:", verdict)
    m = X._mlflow()
    with m.start_run(run_name=f"s26 stacking check {datetime.date.today()}"):
        m.set_tags({"phase": "s26-stack", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "verdict": verdict,
                    "weights": str(dict(zip(fams, np.round(coef, 3)))), "rule": "replace ET+XGB only if even RMSE < 14.62"})
        m.log_metrics({f"{k}_{s}": v for k, d in out.items() for s, v in d.items()})
