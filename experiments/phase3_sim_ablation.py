"""Phase 3a: one-factor ablations of the new simulator options; selection on dev, reported on val."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlflow_experiments as X
from f1_weekend_model import ModelParams
from experiments.parallel import eval_many

if __name__ == "__main__":
    mlflow = X._mlflow()
    res, truth, ctruth = X.get_data()
    base = {}
    variants = [("default", {})]
    variants += [(f"team_block={b} link={l}", {"team_block": b, "block_link": l}) for b in (4, 6, 8, 12) for l in (2.0, 4.0, 8.0)]
    variants += [(f"team_block=6 link=4 half_life={h}", {"team_block": 6, "block_link": 4.0, "half_life": h}) for h in (15.0, 25.0)]
    variants += [(f"p0_max={v}", {"p0_max": v}) for v in (0.22, 0.30)]
    variants += [(f"dnf_a_team={v}", {"dnf_a_team": v}) for v in (8.0, 40.0)]
    variants += [(f"dnf_a_driver={v}", {"dnf_a_driver": v}) for v in (2.0, 15.0)]
    variants += [(f"dnf_half_life_mult={v}", {"dnf_half_life_mult": v}) for v in (1.0, 4.0)]
    with mlflow.start_run(run_name=f"phase3a-sim-ablation {datetime.date.today()}"):
        mlflow.set_tags({"phase": "3a-ablation", "git_sha": X.git_sha()})
        for split in ("dev", "val"):
            outs = eval_many([v for _, v in variants], split)
            ref = ("sim_default", outs[0][1])
            for (name, v), (fc, pw, secs) in zip(variants, outs):
                _, _, s = X.run_and_log(f"abl {name} [{split}]", X.Sim(ModelParams(**v)), split, ref=ref,
                                        tags={"phase": "3a", "variant": name}, fc_override=fc)
                print(f"{split} {name:38s} MAE {s['mae']:.3f} rho {s['spearman']:.3f} dnf_brier {s.get('dnf_brier', float('nan')):.4f} cov80 {s.get('cover80', float('nan')):.3f}")
