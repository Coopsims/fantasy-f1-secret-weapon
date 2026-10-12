"""Seed averaging: does averaging several random seeds of each tree model help, and how much does one seed move the
forecast?  Specs = the interleaved run's (tuned on odd 2026 gamedays); scored on the even ones (never tuned on).
Pre-registered rule: deploy N seeds if the seed-averaged ensemble's even-gameday RMSE is not worse than the mean
single-seed ensemble RMSE (it is a variance reduction - the question is mainly how large the seed noise is)."""
import os, sys, datetime, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

CACHE, PANEL, LOG = os.environ["S26_CACHE"], os.environ["PANEL_PKL"], os.environ["SPECS_LOG"]
SEEDS = list(range(int(os.environ.get("N_SEEDS", "5"))))
FAMS = os.environ.get("FAMS", "extratrees,xgb").split(",")


def _run(args):
    fam, seed, spec = args
    from experiments import s26
    D = s26.prepare(PANEL, CACHE)
    return fam, seed, s26.predict_spec(D, {**spec, "seed": seed}, D["ts26"])


if __name__ == "__main__":
    import mlflow_experiments as X
    from experiments import s26
    from experiments.s26_search import _specs_from_log
    D = s26.prepare(PANEL, CACHE)
    specs = _specs_from_log(LOG)
    even, odd = s26.gamedays(D, 1, 17, "even"), s26.gamedays(D, 1, 17, "odd")
    with ProcessPoolExecutor(max_workers=5) as ex:
        out = list(ex.map(_run, [(f, s, specs[f]) for f in FAMS for s in SEEDS]))
    P = {(f, s): p for f, s, p in out}
    ens = lambda parts: pd.concat(parts).groupby(["t", "driver"], as_index=False)["pred"].mean()
    rows = []

    def add(name, pred, **kw):
        for split, ts in (("odd", odd), ("even", even)):
            pw = s26.score(D, pred[pred["t"].isin(ts)])
            rows.append({"name": name, "split": split, **s26.summary(pw), **kw})
        return s26.score(D, pred[pred["t"].isin(even)])

    single = {}
    for s in SEEDS:
        for f in FAMS:
            add(f"{f} seed {s}", P[(f, s)], seed=s, n_seeds=1)
        single[s] = add(f"ens seed {s}", ens([P[(f, s)] for f in FAMS]), seed=s, n_seeds=1)
    for f in FAMS:
        add(f"{f} {len(SEEDS)}-seed avg", ens([P[(f, s)] for s in SEEDS]), n_seeds=len(SEEDS))
    avg = ens([P[(f, s)] for f in FAMS for s in SEEDS])
    pw_avg = add(f"ens {len(SEEDS)}-seed avg", avg, n_seeds=len(SEEDS))
    df = pd.DataFrame(rows)
    # how far one seed moves a single driver's forecast (sd across seeds, per driver-weekend), and the top-5 overlap
    wide = pd.concat([ens([P[(f, s)] for f in FAMS]).set_index(["t", "driver"])["pred"].rename(s) for s in SEEDS], axis=1)
    sd = float(wide.std(axis=1).mean())
    top5 = []
    for t, g in wide.groupby(level=0):
        sets = [set(g[s].nlargest(5).index.get_level_values(1)) for s in SEEDS]
        top5.append(np.mean([len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]))
    ev = df[df.split == "even"].set_index("name")
    single_mean = float(ev.loc[[f"ens seed {s}" for s in SEEDS], "rmse"].mean())
    bt = X.paired_bootstrap(pw_avg, single[0], "rmse")
    verdict = "adopt" if ev.loc[f"ens {len(SEEDS)}-seed avg", "rmse"] <= single_mean else "reject"
    print(df.pivot_table(index="name", columns="split", values=["rmse", "pick5_ratio"]).round(3).to_string())
    print(f"\neven: single-seed ensemble RMSE mean {single_mean:.3f} (range {ev.loc[[f'ens seed {s}' for s in SEEDS], 'rmse'].min():.3f}"
          f"-{ev.loc[[f'ens seed {s}' for s in SEEDS], 'rmse'].max():.3f}); {len(SEEDS)}-seed avg {ev.loc[f'ens {len(SEEDS)}-seed avg', 'rmse']:.3f}; "
          f"vs seed 0 {bt['delta']:+.3f} [{bt['lo']:+.3f}, {bt['hi']:+.3f}]")
    print(f"seed sd of a driver forecast {sd:.2f} pts; mean top-5 overlap between seeds {np.mean(top5):.2f}/5; verdict: {verdict}")
    m = X._mlflow()
    with m.start_run(run_name=f"s26 seed averaging {datetime.date.today()}"):
        m.set_tags({"phase": "s26-seed-avg", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "families": ",".join(FAMS),
                    "specs_from": LOG, "rule": "adopt if avg RMSE (even) <= mean single-seed RMSE (even)", "verdict": verdict})
        m.log_metrics({"even_rmse_avg": float(ev.loc[f"ens {len(SEEDS)}-seed avg", "rmse"]), "even_rmse_single_mean": single_mean,
                       "seed_sd_points": sd, "top5_overlap": float(np.mean(top5)), "vs_seed0_delta": bt["delta"],
                       "vs_seed0_lo": bt["lo"], "vs_seed0_hi": bt["hi"], "n_seeds": len(SEEDS)})
        df.to_csv("results/s26_seed_avg.csv", index=False)
        m.log_artifact("results/s26_seed_avg.csv")
