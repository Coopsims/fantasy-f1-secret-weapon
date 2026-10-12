"""
Model zoo (experiments/zoo.py)
==============================
Extended pre-weekend feature panel + a registry of model families with default settings and Optuna search spaces,
evaluated walk-forward (training rows strictly before the target weekend; features known before the team lock).

Feature groups (all computed from earlier weekends, or from this weekend's sessions that start before the lock):
  form      d_last1/3/5/10, EWMAs at 3 speeds, volatility, quali/race position EWMAs, DNF rate, teammate gap, Elo
  team      team form (EWMA, last-3), team quali EWMA
  track     hand profiles (TRACK_PROFILES), street flag, circuit history (overtakes, grid stickiness, DNF rate,
            safety cars), practice speed index, similar-track affinity of driver and team
  weekend   practice pace (f1_practice), practice speed traps (driver, team), practice weather, Sprint Qualifying
            result (Sprint weekends, known at the lock), sprint flag, round in season
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

import mlflow_experiments as X
import f1_practice as P
from f1_calendar import canon_location

STREET = {"Monaco", "Baku", "Marina Bay", "Jeddah", "Las Vegas", "Miami", "Melbourne", "Montréal", "Madrid"}


# ──────────────────────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────────────────────

def elo(truth, k=24.0):
    """Pre-weekend multi-player Elo for race order and qualifying order."""
    rr, rq, rows = {}, {}, []
    for t, g in truth.sort_values("t").groupby("t", sort=True):
        names = list(g["driver"])
        rows += [(t, d, rr.get(d, 1500.0), rq.get(d, 1500.0)) for d in names]
        race = np.where(g["race_dnf"].values, 30.0, g["race_pos"].astype(float).values)
        quali = g["quali_pos"].astype(float).replace(0, np.nan).values
        for pos, R in ((race, rr), (quali, rq)):
            ok = ~np.isnan(pos)
            nm = [n for n, o in zip(names, ok) if o]
            p = pos[ok]
            if len(nm) < 4:
                continue
            r = np.array([R.get(n, 1500.0) for n in nm])
            E = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))
            S = (p[:, None] < p[None, :]).astype(float) + 0.5 * (p[:, None] == p[None, :])
            np.fill_diagonal(E, 0); np.fill_diagonal(S, 0)
            delta = k / (len(nm) - 1) * (S - E).sum(1)
            for n, v in zip(nm, r + delta):
                R[n] = float(v)
    return pd.DataFrame(rows, columns=["t", "driver", "elo_race", "elo_quali"])


def _weekend_tables(full, weekends):
    """From the one-pull cache: per weekend t -> track stats, practice speeds/weather; per (t, driver) practice speed."""
    from openf1_cache import lock_time
    key = {(int(r.year), r.location): int(r.t) for r in weekends.itertuples()}
    wk, drv = [], []
    for e in full:
        m = e["meeting"]
        t = key.get((int(m["year"]), canon_location(m["location"])))
        if t is None:
            continue
        lock = lock_time(e["sessions"])
        name = {s["session_key"]: s["session_name"] for s in e["sessions"]}
        start = {s["session_key"]: s["date_start"] for s in e["sessions"]}
        pre = {sk for sk, nm in name.items() if nm.startswith("Practice") and lock and start[sk] < lock}
        race = {sk for sk, nm in name.items() if nm == "Race"}
        laps = pd.DataFrame(e["endpoints"].get("laps") or [])
        dn = {(d["session_key"], d["driver_number"]): d.get("last_name", "") for d in e["endpoints"].get("drivers") or []}
        row = {"t": t}
        if len(laps):
            fp = laps[laps["session_key"].isin(pre) & laps["st_speed"].notna()]
            row["speed_fp"] = float(fp["st_speed"].median()) if len(fp) else np.nan
            rl = laps[laps["session_key"].isin(race) & laps["st_speed"].notna()]
            row["speed_race"] = float(rl["st_speed"].median()) if len(rl) else np.nan
            if len(fp):
                from f1_calendar import canon_driver
                q = fp.groupby(["session_key", "driver_number"])["st_speed"].quantile(0.9).reset_index()
                q["driver"] = [canon_driver(dn.get((sk, n), "")) for sk, n in zip(q["session_key"], q["driver_number"])]
                q = q.groupby("driver")["st_speed"].mean() - fp["st_speed"].quantile(0.9)
                drv += [(t, d, float(v)) for d, v in q.items() if d]
        rc = [r for r in e["endpoints"].get("race_control") or [] if r.get("session_key") in race]
        row["sc"] = sum(1 for r in rc if r.get("category") == "SafetyCar" and "DEPLOYED" in (r.get("message") or ""))
        w = pd.DataFrame(e["endpoints"].get("weather") or [])
        if len(w):
            wf = w[w["session_key"].isin(pre)]
            row["fp_track_temp"] = float(wf["track_temperature"].mean()) if len(wf) else np.nan
            row["fp_rain"] = float(wf["rainfall"].astype(float).max()) if len(wf) else np.nan
        wk.append(row)
    return pd.DataFrame(wk), pd.DataFrame(drv, columns=["t", "driver", "d_fp_speed"])


def _past_circuit_stats(truth, wk):
    """Circuit history from earlier visits only: overtakes, stickiness, DNF rate, safety cars, race speed."""
    g = truth.groupby("t")
    per = pd.DataFrame({"t": list(g.groups),
                        "ot": g.apply(lambda x: np.nan).values})
    rows = []
    for t, x in g:
        fin = x[~x["race_dnf"]]
        stick = fin[["quali_pos", "race_pos"]].replace(0, np.nan).dropna().corr(method="spearman").iloc[0, 1] if len(fin) > 5 else np.nan
        rows.append((t, x["location"].iloc[0], x["race_dnf"].mean(), stick))
    per = pd.DataFrame(rows, columns=["t", "location", "dnf", "stick"]).merge(wk[["t", "sc", "speed_race"]], on="t", how="left")
    ot = truth.groupby("t").apply(lambda x: np.nan)
    per = per.sort_values("t")
    out = []
    for r in per.itertuples():
        past = per[(per["location"] == r.location) & (per["t"] < r.t)]
        out.append((r.t, past["dnf"].mean(), past["stick"].mean(), past["sc"].mean(), past["speed_race"].mean(), len(past)))
    return pd.DataFrame(out, columns=["t", "trk_hist_dnf", "trk_hist_stick", "trk_hist_sc", "trk_hist_speed", "trk_visits"])


def _affinity(panel, speed, key, value, bandwidth=12.0, half_life=25.0):
    """Kernel-weighted mean of a past z-score at tracks of similar practice speed, minus the plain past mean."""
    out = np.full(len(panel), np.nan)
    sp = speed.set_index("t")["speed_fp"]
    tt = panel["t"].values
    for k, idx in panel.groupby(key).indices.items():
        ts, vs = tt[idx], panel[value].values[idx]
        order = np.argsort(ts, kind="stable")
        for jj, j in enumerate(order):
            t = ts[j]
            past = order[:jj]
            past = past[ts[past] < t]               # strictly earlier weekends: a team has 2 rows per weekend
            if len(past) < 5 or pd.isna(sp.get(t)):
                continue
            s_past = np.array([sp.get(ts[p], np.nan) for p in past])
            ok = ~np.isnan(s_past) & ~np.isnan(vs[past])
            if ok.sum() < 5:
                continue
            decay = 0.5 ** ((t - ts[past][ok]) / half_life)
            w = decay * np.exp(-((s_past[ok] - sp[t]) / bandwidth) ** 2)
            if w.sum() <= 1e-9:
                continue
            out[idx[j]] = float((w * vs[past][ok]).sum() / w.sum() - (decay * vs[past][ok]).sum() / decay.sum())
    return out


def _sim_affinity(df, S, key, value, half_life=25.0, groups=None):
    """Past-only: similarity-weighted (or same-group) mean of a weekend z-score minus the plain decayed mean."""
    out = np.full(len(df), np.nan)
    tt, loc, val = df["t"].values, df["location"].values, df[value].values
    for _, idx in df.groupby(key).indices.items():
        order = idx[np.argsort(tt[idx], kind="stable")]
        for jj in range(5, len(order)):
            j, past = order[jj], order[:jj]
            past = past[tt[past] < tt[j]]           # strictly earlier weekends (teammate's row of the same weekend excluded)
            if loc[j] not in S.index:
                continue
            ok = ~np.isnan(val[past]) & np.isin(loc[past], S.index)
            if ok.sum() < 5:
                continue
            p = past[ok]
            decay = 0.5 ** ((tt[j] - tt[p]) / half_life)
            if groups is None:
                w = decay * S.loc[loc[j], loc[p]].values
            else:
                w = decay * (groups.reindex(loc[p]).values == groups.get(loc[j], -1))
            if w.sum() <= 1e-6:
                continue
            out[j] = float((w * val[p]).sum() / w.sum() - (decay * val[p]).sum() / decay.sum())
    return out


def track_tables():
    """Circuit profiles, groups (k chosen by silhouette) and similarity from f1_tracks (no results used)."""
    import f1_tracks as T
    prof = T.circuit_profiles()
    groups, sil, k = T.cluster(prof)
    return prof, groups, T.similarity(prof), sil, k


def build_panel(res, truth, full=None, pdata=None, tracks=None):
    df = X._panel_features(truth)
    g = df.sort_values(["driver", "t"]).groupby("driver", sort=False)["total"]
    extra = pd.DataFrame({"t": df.sort_values(["driver", "t"])["t"].values,
                          "driver": df.sort_values(["driver", "t"])["driver"].values,
                          "d_last10": g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean()).values,
                          "d_ewm_fast": g.transform(lambda s: s.shift(1).ewm(alpha=0.5, adjust=False).mean()).values,
                          "d_ewm_slow": g.transform(lambda s: s.shift(1).ewm(alpha=0.1, adjust=False).mean()).values,
                          "d_std5": g.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std()).values})
    df = df.merge(extra, on=["t", "driver"], how="left")
    mate = df.groupby(["t", "team"])["d_ewm"].transform("sum") - df["d_ewm"]
    nm = df.groupby(["t", "team"])["d_ewm"].transform("count") - 1
    df["mate_gap"] = df["d_ewm"] - mate / nm.replace(0, np.nan)
    tq = df.groupby(["team", "t"], as_index=False)["quali_pos"].mean().sort_values(["team", "t"])
    tq["team_q_ewm"] = tq.groupby("team")["quali_pos"].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean())
    df = df.merge(tq[["team", "t", "team_q_ewm"]], on=["team", "t"], how="left")
    df = df.merge(elo(truth), on=["t", "driver"], how="left")
    df["elo_race_rel"] = df["elo_race"] - df.groupby("t")["elo_race"].transform("mean")
    df["elo_quali_rel"] = df["elo_quali"] - df.groupby("t")["elo_quali"].transform("mean")
    wk_order = res.weekends.sort_values("t").groupby("year").cumcount() + 1
    df = df.merge(pd.DataFrame({"t": res.weekends.sort_values("t")["t"].values, "wk_in_season": wk_order.values}), on="t", how="left")
    df["trk_street"] = df["location"].isin(STREET).astype(float)
    if pdata is not None:
        pt = P.build_practice_features(pdata, res.weekends, res.sessions, extra_field=truth[["t", "driver", "team"]])
        df = df.merge(pt, on=["t", "driver"], how="left")
    if full is not None:
        wk, dsp = _weekend_tables(full, res.weekends)
        df = df.merge(wk[["t", "speed_fp", "fp_track_temp", "fp_rain"]], on="t", how="left")
        df = df.merge(_past_circuit_stats(truth, wk), on="t", how="left")
        df = df.merge(dsp, on=["t", "driver"], how="left")
        df["team_fp_speed"] = df.groupby(["t", "team"])["d_fp_speed"].transform("mean")
        z = df.groupby("t")["total"].transform(lambda s: (s - s.mean()) / (s.std() or 1.0))
        df["_z"] = z
        df["_tz"] = df.groupby(["t", "team"])["_z"].transform("mean")
        df["aff_driver"] = _affinity(df, wk, "driver", "_z")
        df["aff_team"] = _affinity(df, wk, "team", "_tz")
        if tracks is not None:
            prof, groups, S = tracks[:3]
            for c, nm in (("v_mean", "trk_v_mean"), ("full_throttle", "trk_full_throttle"), ("braking", "trk_braking"),
                          ("slow_per_km", "trk_slow"), ("fast_per_km", "trk_fast"), ("turns_per_km", "trk_turns_per_km"),
                          ("length_km", "trk_length")):
                df[nm] = df["location"].map(prof[c])
            for g in sorted(groups.unique()):
                df[f"trk_group_{g}"] = (df["location"].map(groups) == g).astype(float)
            df["aff_driver_sim"] = _sim_affinity(df, S, "driver", "_z")
            df["aff_team_sim"] = _sim_affinity(df, S, "team", "_tz")
            df["aff_driver_grp"] = _sim_affinity(df, S, "driver", "_z", groups=groups)
            df["aff_team_grp"] = _sim_affinity(df, S, "team", "_tz", groups=groups)
        df = df.drop(columns=["_z", "_tz"])
    return df.sort_values(["t", "driver"]).reset_index(drop=True)


FEATURES = {
    "form": ["d_last1", "d_last3", "d_last5", "d_last10", "d_ewm", "d_ewm_fast", "d_ewm_slow", "d_std5", "d_cnt",
             "d_qpos_ewm", "d_rpos_ewm", "d_dnf_ewm", "mate_gap", "elo_race_rel", "elo_quali_rel"],
    "team": ["team_ewm", "team_last3", "team_q_ewm"],
    "track": ["trk_stickiness", "trk_chaos", "trk_overtaking", "trk_stops", "trk_street", "d_track_mean",
              "trk_hist_dnf", "trk_hist_stick", "trk_hist_sc", "trk_hist_speed", "trk_visits", "speed_fp",
              "aff_driver", "aff_team", "trk_v_mean", "trk_full_throttle", "trk_braking", "trk_slow", "trk_fast",
              "trk_turns_per_km", "trk_length", "trk_group_0", "trk_group_1", "trk_group_2",
              "aff_driver_sim", "aff_team_sim", "aff_driver_grp", "aff_team_grp"],
    "weekend": P.FP_FEATURES + P.SQ_FEATURES + ["d_fp_speed", "team_fp_speed", "fp_track_temp", "fp_rain",
                                               "is_sprint", "wk_in_season"],
}
FEATURES["base"] = X.GBM_FEATURES
FEATURES["all"] = FEATURES["form"] + FEATURES["team"] + FEATURES["track"] + FEATURES["weekend"]
FEATURES["no_track"] = FEATURES["form"] + FEATURES["team"] + FEATURES["weekend"]
FEATURES["no_weekend"] = FEATURES["form"] + FEATURES["team"] + FEATURES["track"]


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────

def make_model(family, params=None, seed=0):
    """sklearn-style regressor; params override defaults."""
    p = dict(params or {})
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
    if family == "ridge":
        from sklearn.linear_model import Ridge
        return make_pipeline(*pre(), Ridge(alpha=p.get("alpha", 10.0)))
    if family == "lasso":
        from sklearn.linear_model import Lasso
        return make_pipeline(*pre(), Lasso(alpha=p.get("alpha", 0.1), max_iter=20000))
    if family == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return make_pipeline(*pre(), ElasticNet(alpha=p.get("alpha", 0.1), l1_ratio=p.get("l1_ratio", 0.5), max_iter=20000))
    if family == "huber":
        from sklearn.linear_model import HuberRegressor
        return make_pipeline(*pre(), HuberRegressor(epsilon=p.get("epsilon", 1.5), alpha=p.get("alpha", 1e-3), max_iter=2000))
    if family == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return make_pipeline(*pre(), KNeighborsRegressor(n_neighbors=int(p.get("n_neighbors", 40)), weights=p.get("weights", "distance")))
    if family == "svr":
        from sklearn.svm import SVR
        return make_pipeline(*pre(), SVR(C=p.get("C", 10.0), epsilon=p.get("epsilon", 2.0), gamma=p.get("gamma", "scale")))
    if family == "mlp":
        from sklearn.neural_network import MLPRegressor
        h = int(p.get("width", 32))
        return make_pipeline(*pre(), MLPRegressor(hidden_layer_sizes=(h,) * int(p.get("depth", 2)), alpha=p.get("alpha", 1e-2),
                                                  learning_rate_init=p.get("lr", 1e-3), max_iter=600, early_stopping=True,
                                                  random_state=seed))
    if family == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=int(p.get("n_estimators", 400)), min_samples_leaf=int(p.get("min_samples_leaf", 10)),
            max_features=p.get("max_features", 0.5), n_jobs=4, random_state=seed))
    if family == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=int(p.get("n_estimators", 400)), min_samples_leaf=int(p.get("min_samples_leaf", 10)),
            max_features=p.get("max_features", 0.5), n_jobs=4, random_state=seed))
    if family == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p.get("max_iter", 200)), learning_rate=p.get("learning_rate", 0.05),
                                             max_depth=p.get("max_depth", 3), min_samples_leaf=int(p.get("min_samples_leaf", 20)),
                                             l2_regularization=p.get("l2", 1.0), random_state=seed)
    if family == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=int(p.get("n_estimators", 100)), max_depth=int(p.get("max_depth", 2)),
                            learning_rate=p.get("learning_rate", 0.03), min_child_weight=p.get("min_child_weight", 5),
                            subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.8),
                            reg_lambda=p.get("reg_lambda", 1.0), objective="reg:squarederror", random_state=seed, n_jobs=4)
    if family == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=int(p.get("n_estimators", 200)), num_leaves=int(p.get("num_leaves", 7)),
                             learning_rate=p.get("learning_rate", 0.03), min_child_samples=int(p.get("min_child_samples", 30)),
                             subsample=p.get("subsample", 0.8), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.8),
                             reg_lambda=p.get("reg_lambda", 1.0), random_state=seed, n_jobs=4, verbose=-1)
    if family == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=int(p.get("iterations", 300)), depth=int(p.get("depth", 4)),
                                 learning_rate=p.get("learning_rate", 0.05), l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
                                 random_seed=seed, verbose=False, thread_count=4)
    raise ValueError(family)


SPACES = {   # Optuna search spaces: name -> (kind, low, high[, log])
    "ridge": {"alpha": ("float", 0.01, 1000, True)},
    "elasticnet": {"alpha": ("float", 1e-3, 10, True), "l1_ratio": ("float", 0.05, 0.95)},
    "huber": {"epsilon": ("float", 1.1, 3.0), "alpha": ("float", 1e-5, 10, True)},
    "knn": {"n_neighbors": ("int", 5, 150), "weights": ("cat", ["uniform", "distance"])},
    "svr": {"C": ("float", 0.1, 300, True), "epsilon": ("float", 0.1, 10, True)},
    "mlp": {"width": ("int", 8, 128, True), "depth": ("int", 1, 3), "alpha": ("float", 1e-5, 1.0, True), "lr": ("float", 1e-4, 1e-2, True)},
    "rf": {"n_estimators": ("int", 200, 800), "min_samples_leaf": ("int", 2, 60, True), "max_features": ("float", 0.1, 1.0)},
    "extratrees": {"n_estimators": ("int", 200, 800), "min_samples_leaf": ("int", 2, 60, True), "max_features": ("float", 0.1, 1.0)},
    "hgb": {"max_iter": ("int", 50, 600, True), "learning_rate": ("float", 0.01, 0.2, True), "max_depth": ("int", 2, 6),
            "min_samples_leaf": ("int", 5, 100, True), "l2": ("float", 1e-3, 30, True)},
    "xgb": {"n_estimators": ("int", 50, 800, True), "max_depth": ("int", 1, 6), "learning_rate": ("float", 0.005, 0.2, True),
            "min_child_weight": ("float", 1, 50, True), "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.3, 1.0),
            "reg_lambda": ("float", 0.01, 50, True)},
    "lgbm": {"n_estimators": ("int", 50, 800, True), "num_leaves": ("int", 3, 31), "learning_rate": ("float", 0.005, 0.2, True),
             "min_child_samples": ("int", 5, 120, True), "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.3, 1.0),
             "reg_lambda": ("float", 0.01, 50, True)},
    "catboost": {"iterations": ("int", 100, 1000, True), "depth": ("int", 2, 7), "learning_rate": ("float", 0.01, 0.2, True),
                 "l2_leaf_reg": ("float", 0.5, 30, True)},
}


def suggest(trial, family):
    p = {}
    for k, spec in SPACES[family].items():
        if spec[0] == "float":
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=len(spec) > 3 and spec[3])
        elif spec[0] == "int":
            p[k] = trial.suggest_int(k, spec[1], spec[2], log=len(spec) > 3 and spec[3])
        else:
            p[k] = trial.suggest_categorical(k, spec[1])
    return p


# ──────────────────────────────────────────────────────────────
# Walk-forward evaluation
# ──────────────────────────────────────────────────────────────

def walk_forward(panel, feats, family, params, target_ts, refit_every=1, min_rows=150, seed=0, target="total"):
    """Fit on rows with t < first target of each block, predict the block. Returns Forecast(drivers)."""
    ts = sorted(target_ts)
    rows = []
    for b in range(0, len(ts), refit_every):
        block = ts[b:b + refit_every]
        tr = panel[(panel["t"] < block[0]) & panel["d_last1"].notna()]
        if len(tr) < min_rows:
            continue
        m = make_model(family, params, seed)
        m.fit(tr[feats], tr[target])
        te = panel[panel["t"].isin(block)]
        rows += list(zip(te["t"], te["driver"], m.predict(te[feats]).astype(float)))
    return X.Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"]))


def team_bonus(truth, ctruth, alpha=0.3):
    """Past-only EWMA of (constructor total - sum of its drivers' totals): quali bonus, pit points, etc."""
    s = truth.groupby(["t", "team"])["total"].sum().rename("dsum").reset_index()
    c = ctruth.merge(s, on=["t", "team"])
    c["extra"] = c["total"] - c["dsum"]
    c = c.sort_values(["team", "t"])
    c["bonus"] = c.groupby("team")["extra"].transform(lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
    return c[["t", "team", "bonus"]]


def with_constructors(fc, truth, bonus):
    d = fc.drivers.merge(truth[["t", "driver", "team"]], on=["t", "driver"])
    c = d.groupby(["t", "team"], as_index=False)["pred"].sum().merge(bonus, on=["t", "team"], how="left")
    c["pred"] = c["pred"] + c["bonus"].fillna(0.0)
    return X.Forecast(fc.drivers, c[["t", "team", "pred"]])


def metrics(fc, truth, ctruth):
    """Per-weekend frame with mae, rmse, spearman, top5, pick5 (actual points of the predicted top 5 / best possible)."""
    pw = X.per_weekend(fc, truth, ctruth)
    d = fc.drivers.merge(truth[["t", "driver", "total"]], on=["t", "driver"])
    extra = []
    for t, g in d.groupby("t"):
        best = g.nlargest(5, "total")["total"].sum()
        got = g.nlargest(5, "pred")["total"].sum()
        extra.append((t, float(np.sqrt(((g["pred"] - g["total"]) ** 2).mean())), float(got), float(got / best) if best > 0 else np.nan))
    return pw.merge(pd.DataFrame(extra, columns=["t", "rmse", "pick5_pts", "pick5_ratio"]), on="t")


def summary(pw):
    s = X.summarize(pw)
    for k in ("rmse", "pick5_pts", "pick5_ratio"):
        s[k] = float(pw[k].mean())
    return s
