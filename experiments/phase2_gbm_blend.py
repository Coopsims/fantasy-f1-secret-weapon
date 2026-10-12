"""Phase 2: tune the GBM on dev (2024) only; blend with the simulator; weight chosen on dev, checked on val (2025)."""
import os, sys, datetime, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mlflow_experiments as X

mlflow = X._mlflow()
res, truth, ctruth = X.get_data()
with mlflow.start_run(run_name=f"phase2-gbm-blend {datetime.date.today()}"):
    mlflow.set_tags({"phase": "2-gbm-blend", "git_sha": X.git_sha()})
    sim = {s: X.run_and_log(f"sim_default [{s}]", X.Sim(), s, tags={"phase": "2"}) for s in ("dev", "val")}
    ref = {s: ("sim_default", sim[s][1]) for s in sim}

    # --- GBM grid, selected on dev only ---
    grid = list(itertools.product([2, 3, 4], [100, 200, 400], [0.03, 0.06], ["reg:squarederror", "reg:absoluteerror"]))
    dev_scores = []
    for depth, n, lr, obj in grid:
        m = X.GBM(n_estimators=n, max_depth=depth, learning_rate=lr, objective=obj)
        fc, pw, s = X.run_and_log(f"gbm d{depth} n{n} lr{lr} {obj[4:7]} [dev]", m, "dev", ref=ref["dev"], tags={"phase": "2", "stage": "gbm-grid"})
        dev_scores.append((s["mae"], (depth, n, lr, obj)))
    dev_scores.sort()
    print("top dev GBM configs:", [(round(a, 3), c) for a, c in dev_scores[:5]])
    best = dev_scores[0][1]
    gbm = X.GBM(n_estimators=best[1], max_depth=best[0], learning_rate=best[2], objective=best[3])
    gfc = {s: X.run_and_log(f"gbm_dev_selected [{s}]", gbm, s, ref=ref[s], tags={"phase": "2", "stage": "gbm-selected"}) for s in ("dev", "val")}
    print("selected", best, {s: round(gfc[s][2]["mae"], 3) for s in gfc})

    # --- blend sim + selected GBM; weight on sim chosen on dev ---
    ws = np.round(np.arange(0, 1.01, 0.1), 2)
    curve = {s: {} for s in ("dev", "val")}
    for s in ("dev", "val"):
        for w in ws:
            fc = X.blend(sim[s][0], gfc[s][0], float(w))
            _, pw, summ = X.run_and_log(f"blend sim*{w:.1f}+gbm [{s}]", X.Sim(), s, ref=ref[s],
                                        tags={"phase": "2", "stage": "blend", "w_sim": str(w)}, fc_override=fc)
            curve[s][float(w)] = summ["mae"]
    w_star = min(curve["dev"], key=curve["dev"].get)
    print("blend MAE curve dev:", {k: round(v, 3) for k, v in curve["dev"].items()})
    print("blend MAE curve val:", {k: round(v, 3) for k, v in curve["val"].items()})
    print(f"w_sim chosen on dev = {w_star}; val MAE at that weight {curve['val'][w_star]:.3f} (sim {curve['val'][1.0]:.3f}, gbm {curve['val'][0.0]:.3f})")
    mlflow.log_metrics({"w_sim_dev_selected": w_star, "val_mae_blend_at_dev_w": curve["val"][w_star],
                        "val_mae_sim": curve["val"][1.0], "val_mae_gbm": curve["val"][0.0]})
