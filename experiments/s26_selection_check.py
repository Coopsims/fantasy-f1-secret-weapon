"""Honest estimate for the ensemble CHOICE, not just the tuning.  The deployed pair mean(extra-trees, XGBoost) was
picked after looking at the even (held-out) gamedays, so its 14.62 is optimistic.  Rule fixed before this run
(2026-10-12 01:10): the deployed ensemble = the subset of at most 3 tuned families with the lowest RMSE on the ODD
gamedays (where the families were tuned); its EVEN-gameday RMSE is the honest number; deploy it if it is not ET+XGB.
Inputs: one or more results/s26_<tag>_best.pkl files (walk-forward predictions of every tuned family)."""
import os, sys, itertools, datetime, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

if __name__ == "__main__":
    import mlflow_experiments as X
    from experiments import s26
    D = s26.prepare(os.environ["PANEL_PKL"], os.environ["S26_CACHE"])
    preds = {}
    for p in sys.argv[1:]:
        b = pd.read_pickle(p)
        preds |= {f: b["preds"][f] for f in b["best"]}
    odd, even = s26.gamedays(D, 1, 17, "odd"), s26.gamedays(D, 1, 17, "even")
    rows = []
    for k in (1, 2, 3):
        for sub in itertools.combinations(sorted(preds), k):
            pr = pd.concat([preds[f] for f in sub]).groupby(["t", "driver"], as_index=False)["pred"].mean()
            r = {"subset": "+".join(sub), "size": k}
            for nm, ts in (("odd", odd), ("even", even)):
                r[f"{nm}_rmse"] = s26.summary(s26.score(D, pr[pr["t"].isin(ts)]))["rmse"]
            rows.append(r)
    df = pd.DataFrame(rows).sort_values("odd_rmse")
    pick = df.iloc[0]
    sim = s26.summary(s26.score(D, D["panel"][D["panel"]["t"].isin(even)][["t", "driver", "sim"]].rename(columns={"sim": "pred"}).dropna()))["rmse"]
    et_xgb = df[df.subset == "extratrees+xgb"].iloc[0] if (df.subset == "extratrees+xgb").any() else None
    print(df.head(12).round(3).to_string(index=False))
    print(f"\nchosen on odd gamedays: {pick.subset} (odd {pick.odd_rmse:.3f}) -> honest even RMSE {pick.even_rmse:.3f}; "
          f"simulator even {sim:.3f}; post-hoc ET+XGB even {et_xgb.even_rmse if et_xgb is not None else float('nan'):.3f}; "
          f"best even of all {len(df)} subsets {df.even_rmse.min():.3f} ({df.loc[df.even_rmse.idxmin(), 'subset']})")
    tag = os.environ.get("RUN_TAG", "selection")
    df.to_csv(f"results/s26_selection_{tag}.csv", index=False)
    m = X._mlflow()
    with m.start_run(run_name=f"s26 ensemble selection check {tag} {datetime.date.today()}"):
        m.set_tags({"phase": "s26-selection", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "inputs": ",".join(sys.argv[1:]),
                    "rule": "subset (<=3 families) with lowest odd-gameday RMSE; even RMSE = honest estimate", "chosen": pick.subset})
        m.log_metrics({"chosen_odd_rmse": float(pick.odd_rmse), "chosen_even_rmse": float(pick.even_rmse), "sim_even_rmse": sim,
                       "n_subsets": len(df), "best_even_any_subset": float(df.even_rmse.min())}
                      | ({"et_xgb_even_rmse": float(et_xgb.even_rmse)} if et_xgb is not None else {}))
        m.log_artifact(f"results/s26_selection_{tag}.csv")
