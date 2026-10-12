"""
2026-season search (experiments/s26.py)
=======================================
Goal: the best forecaster of OFFICIAL 2026 fantasy points for the rest of 2026.

A "spec" = model family + its hyperparameters + feature set + how training rows are weighted (recency half-life,
extra weight on 2026 = the new regulation era) + whether 2026 rows train on official points + blend weights with the
simulator (incl. Driver of the Day) and with a recent-form EWMA of official points.

Every prediction for gameday g uses only rows from earlier weekends.  Tuning objective = mean driver RMSE vs official
points on a set of 2026 gamedays; the nested check tunes on gamedays 1-9 and scores 10-17.
"""
import os, sys, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

import mlflow_experiments as X
from experiments import zoo
from f1_calendar import GRID_2026

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = os.environ.get("S26_FAMILIES", "xgb,catboost,extratrees,rf,lgbm").split(",")
FSETS = ["all", "no_track", "no_weekend"]
_C = {}


def prepare(panel_pkl, cache_pkl, sim_params=None):
    """Panel + official targets + simulator and EWMA forecasts for every 2026 weekend (cached).
    sim_params: optional dict of ModelParams for the simulator column (e.g. tuned on 2026 by s26_sim_tune.py)."""
    if os.path.exists(cache_pkl):
        return pd.read_pickle(cache_pkl)
    if sim_params is None and os.environ.get("S26_SIM_PARAMS"):
        import json
        sim_params = json.load(open(os.environ["S26_SIM_PARAMS"]))
    res, truth, ctruth = X.get_data()
    panel = pd.read_pickle(panel_pkl)
    op = pd.read_csv("fantasy_f1_official_points.csv")
    w26 = res.weekends[res.weekends.year == 2026].sort_values("t")
    gd = {int(t): i + 1 for i, t in enumerate(w26["t"])}                      # t -> gameday
    r2t = dict(zip(w26["round"], w26["t"]))
    od = op[op.type == "driver"].assign(t=lambda d: d["round"].map(r2t))
    oc = op[op.type == "constructor"].assign(t=lambda d: d["round"].map(r2t))
    panel = panel.merge(od[["t", "name", "points"]].rename(columns={"name": "driver", "points": "official"}), on=["t", "driver"], how="left")
    own = panel["team"] == panel["driver"].map(GRID_2026)
    panel.loc[(panel["year"] == 2026) & ~own, "official"] = np.nan             # stand-in weekends: asset scored elsewhere
    panel["regime"] = (panel["year"] >= 2026).astype(float)
    ts26 = sorted(gd)
    tg = res.weekends[res.weekends["t"].isin(ts26)].reset_index(drop=True)
    from f1_weekend_model import ModelParams
    sim = X.Sim(ModelParams(**(sim_params or {})), dotd=True, n_sims=4000).predict(res, truth, ctruth, tg)
    simd = sim.drivers.rename(columns={"pred": "sim"})[["t", "driver", "sim"]]
    simc = sim.constructors.rename(columns={"pred": "sim_c"})
    panel = panel.merge(simd, on=["t", "driver"], how="left")
    # recent form on official points (2026 gamedays before t); before any 2026 data: the reconstructed EWMA
    for a in (0.2, 0.3, 0.45):
        col = f"ew_off_{a}"
        panel[col] = np.nan
        for d, g in panel[panel["year"] == 2026].sort_values("t").groupby("driver"):
            s = g["official"].shift(1).ewm(alpha=a, adjust=False, ignore_na=True).mean()
            panel.loc[g.index, col] = s.values
        panel[col] = panel[col].fillna(panel["d_ewm"])
    out = {"panel": panel, "gameday": gd, "ts26": ts26, "official_c": oc[["t", "name", "points"]].rename(columns={"name": "team"}),
           "sim_c": simc, "truth": truth, "ctruth": ctruth}
    pd.to_pickle(out, cache_pkl)
    return out


def _fit(model, Xtr, y, w):
    from sklearn.pipeline import Pipeline
    import inspect
    last = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if "sample_weight" not in inspect.signature(last.fit).parameters:      # kNN: no row weights (half_life / w26 unused)
        return model.fit(Xtr, y)
    if isinstance(model, Pipeline):
        return model.fit(Xtr, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(Xtr, y, sample_weight=w)


def predict_spec(D, spec, targets, return_models=False):
    """Walk-forward driver forecasts for the target weekends under `spec`. Returns DataFrame t, driver, pred, tree."""
    P = D["panel"]
    feats = zoo.FEATURES[spec["fset"]] + ["regime"]
    rows, models = [], []
    for t in targets:
        tr = P[(P["t"] < t) & P["d_last1"].notna()]
        y = tr["total"].values.copy()
        if spec.get("use_official"):
            off = tr["official"].values
            y = np.where(np.isnan(off), y, off)
        age = t - tr["t"].values
        w = 0.5 ** (age / spec["half_life"]) if spec["half_life"] < 999 else np.ones(len(tr))
        w = w * np.where(tr["regime"].values > 0, spec["w26"], 1.0)
        m = zoo.make_model(spec["family"], spec["hp"], seed=spec.get("seed", 0))
        _fit(m, tr[feats], y, w)
        te = P[P["t"] == t]
        tree = m.predict(te[feats]).astype(float)
        n26 = sum(1 for x in D["ts26"] if x < t)                       # 2026 weekends already raced
        if "b_early" in spec:                                          # simulator weight decays as 2026 data accumulate
            dec = np.exp(-n26 / spec["kappa"])
            b = spec["b_late"] + (spec["b_early"] - spec["b_late"]) * dec
        else:
            b = spec["b_sim"]
        c = spec["c_ewma"]
        sim = te["sim"].fillna(pd.Series(tree, index=te.index)).values
        ew = te[f"ew_off_{spec['ewma_alpha']}"].values
        pred = (1 - b - c) * tree + b * sim + c * ew
        rows += list(zip(te["t"], te["driver"], pred, tree))
        if return_models:
            models.append(m)
    out = pd.DataFrame(rows, columns=["t", "driver", "pred", "tree"])
    return (out, models) if return_models else out


def score(D, pred):
    """Per-weekend metrics vs official points (own-car driver-weekends)."""
    P = D["panel"]
    m = pred.merge(P[["t", "driver", "official"]], on=["t", "driver"]).dropna(subset=["official"])
    rows = []
    for t, g in m.groupby("t"):
        best = g.nlargest(5, "official")["official"].sum()
        got = g.nlargest(5, "pred")["official"].sum()
        rows.append({"t": t, "gameday": D["gameday"][t], "rmse": float(np.sqrt(((g.pred - g.official) ** 2).mean())),
                     "mae": float((g.pred - g.official).abs().mean()), "spearman": X._spearman(g.pred, g.official),
                     "pick5_ratio": float(got / best) if best > 0 else np.nan, "bias": float((g.pred - g.official).mean())})
    return pd.DataFrame(rows)


def summary(pw):
    return {k: float(pw[k].mean()) for k in ("rmse", "mae", "spearman", "pick5_ratio", "bias")} | {"n_weekends": int(len(pw))}


def suggest_spec(trial, family):
    spec = {"family": family, "hp": zoo.suggest(trial, family),
            "fset": trial.suggest_categorical("fset", FSETS),
            "half_life": trial.suggest_categorical("half_life", [6, 12, 25, 50, 100, 999]),
            "w26": trial.suggest_float("w26", 1.0, 10.0, log=True),
            "use_official": trial.suggest_categorical("use_official", [False, True]),
            "ewma_alpha": trial.suggest_categorical("ewma_alpha", [0.2, 0.3, 0.45])}
    spec["b_early"] = trial.suggest_float("b_early", 0.0, 1.0)
    spec["b_late"] = trial.suggest_float("b_late", 0.0, 0.8)
    spec["kappa"] = trial.suggest_float("kappa", 0.5, 12.0, log=True)
    spec["c_ewma"] = trial.suggest_float("c_ewma_raw", 0.0, 0.5) * (1 - max(spec["b_early"], spec["b_late"]))
    return spec


def gamedays(D, lo, hi, parity=None):
    """2026 weekends with gameday in [lo, hi]; parity 'odd' / 'even' keeps every other one (interleaved split)."""
    keep = {"odd": 1, "even": 0}.get(parity)
    return [t for t in D["ts26"] if lo <= D["gameday"][t] <= hi and (keep is None or D["gameday"][t] % 2 == keep)]


def constructor_eval(D, driver_pred, c_sims=(0.0, 0.25, 0.5, 0.75, 1.0), alphas=(0.2, 0.4)):
    """Constructor forecasts = c_sim * simulator + (1 - c_sim) * (sum of its two drivers' forecasts + recent official
    bonus).  Bonus = EWMA over earlier 2026 gamedays of (official constructor points - its drivers' official points).
    Returns per-(c_sim, alpha, gameday) errors vs official constructor points."""
    P = D["panel"]
    oc = D["official_c"].rename(columns={"points": "official"})
    dsum_off = P[P["year"] == 2026].assign(team26=lambda d: d["driver"].map(GRID_2026)).groupby(["t", "team26"])["official"].sum(min_count=1)
    extra = oc.set_index(["t", "team"])["official"] - dsum_off.reindex(oc.set_index(["t", "team"]).index).values
    extra = extra.rename("extra").reset_index().sort_values("t")
    dp = driver_pred.assign(team=lambda d: d["driver"].map(GRID_2026)).groupby(["t", "team"], as_index=False)["pred"].sum()
    simc = D["sim_c"].rename(columns={"pred": "sim_c"})
    rows = []
    for a in alphas:
        b = extra.copy()
        b["bonus"] = b.groupby("team")["extra"].transform(lambda s: s.shift(1).ewm(alpha=a, adjust=False, ignore_na=True).mean())
        m = dp.merge(b[["t", "team", "bonus"]], on=["t", "team"], how="left").merge(simc, on=["t", "team"]).merge(oc, on=["t", "team"])
        m["derived"] = m["pred"] + m["bonus"].fillna(0.0)
        for c in c_sims:
            m["fc"] = c * m["sim_c"] + (1 - c) * m["derived"]
            for t, g in m.groupby("t"):
                rows.append({"alpha": a, "c_sim": c, "t": t, "gameday": D["gameday"][t],
                             "rmse": float(np.sqrt(((g.fc - g.official) ** 2).mean())), "mae": float((g.fc - g.official).abs().mean())})
    return pd.DataFrame(rows)
