"""Write results/s26_final_spec.json from the deployment tuning log (specs tuned on all 17 2026 gamedays) with the
honest evidence from the interleaved run, constructor settings and the planning blend."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("PANEL_PKL", "unused-for-parsing")
from experiments.s26_search import _specs_from_log

if __name__ == "__main__":
    deploy_log, families = sys.argv[1], sys.argv[2].split(",")
    best = _specs_from_log(deploy_log)
    specs = [best[f] for f in families]
    out = {
        "name": f"2026 deploy: mean({', '.join(families)}) tuned on all 17 gamedays (walk-forward, official points)",
        "specs": specs,
        "c_sim": 0.0, "bonus_alpha": 0.2,
        "planning_blend": {"w_model": 0.5, "w_sim": 0.3, "w_ewma": 0.2, "ewma_alpha": 0.2},
        "evidence": ("Honest estimate (same family/structure tuned on odd 2026 gamedays, scored on the 8 even ones, leak-free "
                     "panel): driver RMSE 14.62 vs simulator 15.22 (-0.60, 95% CI -1.26..+0.05), vs simulator+form 15.12, "
                     "vs previous best 15.01; pick-5 value 0.79 vs 0.71. Constructors derived (c_sim 0): RMSE ~20.6 vs simulator "
                     "22.0-22.2 on both halves. Decisions: planning blend 0.5 model + 0.3 simulator + 0.2 form played the 2026 "
                     "season best walk-forward (4151 = 72% of the 5741 optimum; model alone 3953, simulator+form 3955, "
                     "simulator 3330) - in-sample, one season."),
    }
    json.dump(out, open("results/s26_final_spec.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "specs"}, indent=1))
    for s in specs:
        print(s["family"], {k: v for k, v in s.items() if k not in ("hp", "family")}, s["hp"])
