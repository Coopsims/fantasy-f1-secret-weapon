"""Phase 6: use what is known at the team lock (FP1-3 on normal weekends; FP1 + Sprint Qualifying on Sprint weekends).
dev = 2024 selects, val = 2025 confirms, 2026 untouched here (see phase6b)."""
import os, sys, datetime, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from experiments import ens
import mlflow_experiments as X
import f1_practice as P

if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    ptab = P.build_practice_features(P.load_practice(), res.weekends, res.sessions)
    wk = res.weekends[res.weekends["year"] <= 2025]
    allt = wk[[(res.weekends["t"] < t).sum() >= 8 for t in wk["t"]]].reset_index(drop=True)
    cover = ptab.groupby("t")["fp_gap_min"].apply(lambda s: s.notna().mean())
    print(f"practice features cover {len(cover)} weekends; mean share of drivers with a practice lap: {cover.mean():.2f}")
    fp_cols, sq_cols = P.FP_FEATURES, P.SQ_FEATURES
    ts = {"dev": set(allt[allt['year'] == 2024]['t']), "val": set(allt[allt['year'] == 2025]['t'])}
    sprint_t = set(res.weekends[res.weekends["sprint"]]["t"])

    # ---- GBM hyper-parameters for the practice model: small grid, dev only ----
    grid = list(itertools.product([2, 3], [100, 200, 300], [0.03, 0.06]))
    def mk(depth, n, lr, cols): return X.GBM(n_estimators=n, max_depth=depth, learning_rate=lr, extra=ptab, extra_cols=cols)
    dev_t = allt[allt["year"] == 2024]
    sc = []
    for depth, n, lr in grid:
        fc = mk(depth, n, lr, fp_cols + sq_cols).predict(res, truth, ctruth, allt)
        sc.append((X.per_weekend(ens.subset(fc, ts["dev"]), truth)["mae"].mean(), (depth, n, lr)))
    sc.sort(); best = sc[0][1]
    print("practice-GBM grid on dev, best:", [(round(a, 3), c) for a, c in sc[:3]])

    base = {"sim": X.Sim().predict(res, truth, ctruth, allt),
            "sim_sq": X.Sim(known_sq=True).predict(res, truth, ctruth, allt),
            "gbm": X.GBM(**ens.GBM_KW).predict(res, truth, ctruth, allt),
            "gbm_fp": mk(*best, fp_cols).predict(res, truth, ctruth, allt),
            "gbm_fp_sq": mk(*best, fp_cols + sq_cols).predict(res, truth, ctruth, allt)}
    wide = ens.wide_table(truth, base)
    cands = {"sim_default": base["sim"], "sim + known Sprint Quali": base["sim_sq"], "gbm (no practice)": base["gbm"],
             "gbm + practice": base["gbm_fp"], "gbm + practice + SQ": base["gbm_fp_sq"]}
    for a, b in (("sim", "gbm"), ("sim", "gbm_fp_sq"), ("sim_sq", "gbm_fp_sq"), ("sim", "gbm_fp"), ("sim_sq", "gbm_fp")):
        fc, wts = ens.rolling_nnls(wide, [a, b], allt["t"])
        cands[f"nnls({a},{b})"] = fc
    with mlflow.start_run(run_name=f"phase6-practice {datetime.date.today()}"):
        mlflow.set_tags({"phase": "6-practice", "git_sha": X.git_sha(), "gbm_fp_params": str(best)})
        for split in ("dev", "val"):
            ref = prev = None
            for name, fc in cands.items():
                sub = ens.subset(fc, ts[split])
                _, pw, s = X.run_and_log(f"{name} [{split}]", X.Sim(), split, ref=ref, tags={"phase": "6", "cand": name}, fc_override=sub)
                if name == "sim_default": ref = ("sim_default", pw)
                if name == "nnls(sim,gbm)": prev = pw
                bt = X.paired_bootstrap(pw, ref[1], "mae")
                vs_prev = X.paired_bootstrap(pw, prev, "mae") if prev is not None else None
                sp = pw[pw["t"].isin(sprint_t)]["mae"].mean(); ns = pw[~pw["t"].isin(sprint_t)]["mae"].mean()
                print(f"{split} {name:26s} MAE {s['mae']:.3f} (sprint wk {sp:.2f} / normal {ns:.2f}) rho {s['spearman']:.3f} | vs sim {bt['delta']:+.3f} [{bt['lo']:+.3f},{bt['hi']:+.3f}]"
                      + (f" | vs prev-best {vs_prev['delta']:+.3f} [{vs_prev['lo']:+.3f},{vs_prev['hi']:+.3f}]" if vs_prev and name != "nnls(sim,gbm)" else ""))
