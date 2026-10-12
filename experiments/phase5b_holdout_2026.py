"""Phase 5b: ONE-SHOT held-out evaluation on 2026.
Everything below was fixed before any candidate was scored on 2026 (see the pre-registration run):
  primary   nnls(sim, gbm)   - adaptive weights re-fit on earlier weekends only; GBM settings chosen on 2024
  secondary blend 0.4 sim + 0.6 gbm (weight chosen on 2024); sim_default; gbm; ewma(0.2); last3
  rule      adopt the primary if driver-MAE change vs sim_default is < 0 with a 95% bootstrap CI excluding 0;
            otherwise report it as NOT confirmed.
Nothing in the model was changed in response to 2026 data.
"""
import os, sys, datetime, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments import ens
import mlflow_experiments as X

if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    prereg = {"primary": "nnls(sim,gbm)", "secondary": "blend0.4sim+0.6gbm; sim_default; gbm; ewma0.2; last3",
              "gbm_params": json.dumps(ens.GBM_KW), "rule": "adopt iff driver MAE delta vs sim_default <0 with 95% CI excluding 0",
              "model_changes_driven_by_2026": "none", "registered_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "git_sha": X.git_sha(), "data_fingerprint": X.data_fingerprint(res)}
    with mlflow.start_run(run_name="PRE-REGISTRATION 2026 holdout"):
        mlflow.set_tags({"phase": "5b-prereg", **prereg})
    wk = res.weekends
    allt = wk[[(wk["t"] < t).sum() >= 8 for t in wk["t"]]].reset_index(drop=True)      # includes 2026
    base = ens.base_forecasts(res, truth, ctruth, allt)
    wide = ens.wide_table(truth, base)
    nn, wts = ens.rolling_nnls(wide, ["sim", "gbm"], allt["t"])
    fixed = X.blend(base["sim"], base["gbm"], 0.4)
    cands = {"sim_default": base["sim"], "nnls(sim,gbm) [primary]": ens.adjust_constructors(base["sim"], nn, wide),
             "blend 0.4 sim + 0.6 gbm": ens.adjust_constructors(base["sim"], fixed, wide),
             "gbm": base["gbm"], "ewma_0.2": base["ewma"], "last3": base["last3"]}
    ts = set(allt[allt["year"] == 2026]["t"])
    print("2026 weekends scored:", len(ts), "| mean NNLS weights in 2026 (sim, gbm):",
          wts[wts["t"].isin(ts)][["sim", "gbm"]].mean().round(2).to_dict())
    with mlflow.start_run(run_name=f"phase5b-holdout-2026 {datetime.date.today()}"):
        mlflow.set_tags({"phase": "5b-holdout", "git_sha": X.git_sha()})
        ref = None
        for name, fc in cands.items():
            sub = ens.subset(fc, ts)
            _, pw, s = X.run_and_log(f"{name} [test 2026]", X.Sim(), "test", ref=ref, tags={"phase": "5b", "cand": name}, fc_override=sub)
            if name == "sim_default":
                ref = ("sim_default", pw)
                pw_ref = pw
            bt = X.paired_bootstrap(pw, pw_ref, "mae")
            cm = ""
            if "c_mae" in pw and name in ("sim_default", "nnls(sim,gbm) [primary]", "blend 0.4 sim + 0.6 gbm"):
                cb = X.paired_bootstrap(pw.rename(columns={"c_mae": "cm"})[["t", "cm"]], pw_ref.rename(columns={"c_mae": "cm"})[["t", "cm"]], "cm")
                cm = f" | constructor MAE {s['c_mae']:.2f} ({cb['delta']:+.2f} [{cb['lo']:+.2f},{cb['hi']:+.2f}])"
            print(f"{name:28s} driver MAE {s['mae']:.3f} rho {s['spearman']:.3f} top5 {s['top5']:.3f} | vs sim {bt['delta']:+.3f} [{bt['lo']:+.3f},{bt['hi']:+.3f}] P(better) {bt['p_a_lower']:.2f}{cm}")
        wts[wts["t"].isin(ts)].to_csv("/tmp/_unused.csv", index=False) if False else None
