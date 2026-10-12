#!/usr/bin/env python3
"""
Experiment harness (mlflow_experiments.py)
==========================================
Walk-forward evaluation of weekend forecasters, every run logged to MLflow.

Protocol (kept strict so that gains are believable)
  * Every prediction for weekend t uses only weekends strictly before t.
  * Splits:  dev = 2024   (tune here)
             val = 2025   (confirm here, never tuned on)
             test = 2026  (held out: evaluated only through `holdout()`, on finalists)
  * Candidates are compared with a paired bootstrap over weekends, not by eyeballing means.

Forecasters (all produce a mean fantasy-points forecast per driver and weekend)
  Sim      the weekend simulator (ratings + Monte Carlo), optionally with extended ModelParams
  Baseline last-1/3/5 mean, EWMA, season-to-date mean
  GBM      gradient-boosted regression on rolling form / team form / track features
  Blend    convex combination of any two forecasts

View results:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from f1_calendar import track_profile
from f1_weekend_data import Results, load_results, score_weekend_actual
from f1_weekend_model import ModelParams, fit_model, simulate_weekend

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKING_URI = f"sqlite:///{HERE}/mlflow.db"
ARTIFACT_ROOT = f"file://{HERE}/mlartifacts"
EXPERIMENT = "f1-weekend-forecasting"
PROTOCOL = "v1: dev=2024, val=2025, test=2026 held out; paired bootstrap over weekends"
DATA_VERSION = "v2-session-names"   # v1 = 2023-24 Sprint Qualifying/Qualifying swapped, 2023 gaps; see results/DATA_CORRECTION.md
SPLITS = {"dev": 2024, "val": 2025, "test": 2026}


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

def build_truth(res):
    """One row per (weekend, driver): actual fantasy points (no Driver of the Day) and context."""
    frames, cons = [], []
    for t in res.weekends["t"]:
        dd, cd = score_weekend_actual(res, int(t))
        if dd.empty:
            continue
        keep = ["driver", "team", "quali", "sprint", "race", "total"] + \
               [c for c in ("quali_pos", "race_pos", "race_dnf") if c in dd]
        d = dd[keep].copy()
        d["t"] = int(t)
        frames.append(d)
        c = cd[["team", "total", "pit_known"]].copy()
        c["t"] = int(t)
        cons.append(c)
    truth = pd.concat(frames, ignore_index=True)
    if "race_dnf" not in truth:
        truth["race_dnf"] = False
    truth["race_dnf"] = truth["race_dnf"].astype(bool)
    wk = res.weekends[["t", "year", "round", "location", "sprint"]].rename(columns={"sprint": "is_sprint"})
    truth = truth.merge(wk, on="t")
    ctruth = pd.concat(cons, ignore_index=True)
    return truth.sort_values(["t", "driver"]).reset_index(drop=True), ctruth


def data_fingerprint(res):
    h = pd.util.hash_pandas_object(res.sessions[["t", "session", "driver", "finish"]], index=False).values
    return hashlib.sha1(h.tobytes()).hexdigest()[:12]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def targets_for(res, year, min_history=8):
    wk = res.weekends[res.weekends["year"] == year]
    return wk[[(res.weekends["t"] < t).sum() >= min_history for t in wk["t"]]].reset_index(drop=True)


_CACHE = {}


def get_data(historical=None, current=None):
    """Load results + truth once per process."""
    key = (historical, current)
    if key not in _CACHE:
        kw = {}
        if historical:
            kw["historical"] = historical
        if current is not None:
            kw["current"] = current
        res = load_results(verbose=False, **kw)
        truth, ctruth = build_truth(res)
        _CACHE[key] = (res, truth, ctruth)
    return _CACHE[key]


# ──────────────────────────────────────────────────────────────
# Forecasters.  predict(res, truth, ctruth, targets) -> Forecast
# ──────────────────────────────────────────────────────────────

@dataclass
class Forecast:
    drivers: pd.DataFrame            # t, driver, pred [, p10, p90, p_dnf]
    constructors: pd.DataFrame = None  # t, team, pred


class Sim:
    kind = "sim"

    def __init__(self, params=None, n_sims=3000, seed=11, recalibrate_every=6, known_sq=False, dotd=False):
        self.params = replace(params or ModelParams(), dotd=dotd)     # dotd=True when scoring against official points
        self.n_sims, self.seed, self.recal = n_sims, seed, recalibrate_every
        self.known_sq = known_sq       # Sprint weekends: Sprint Qualifying is over before the lock

    def describe(self):
        d = {f"p.{k}": v for k, v in asdict(self.params).items()}
        d.update(n_sims=self.n_sims, seed=self.seed, recalibrate_every=self.recal, known_sq=self.known_sq)
        return d

    def predict(self, res, truth, ctruth, targets):
        p = self.params
        rows, crows = [], []
        calib, since = None, 0
        for w in targets.itertuples():
            t, year = int(w.t), int(w.year)
            past = res.weekends[res.weekends["t"] < t]
            if len(past) < p.min_calib_history:
                continue
            keep = set(past["t"])
            hist = Results(res.sessions[res.sessions["t"].isin(keep)],
                           res.pits[res.pits["t"].isin(keep)] if not res.pits.empty else res.pits, past)
            sess = res.sessions[res.sessions["t"] == t]
            field_ = dict(zip(sess["driver"], sess["team"]))
            race = {"round": int(w.round), "location": w.location, "sprint": bool(w.sprint), "name": w.location}
            if since % self.recal == 0:
                calib = None
            model = fit_model(hist, field_, year, float(t), p, calib)
            calib, since = (model.q_cal, model.r_cal), since + 1
            known = None
            if self.known_sq and bool(w.sprint):
                sq = sess[sess["session"] == "SQ"].sort_values("finish")
                if len(sq):
                    known = {"sprint_quali": list(sq["driver"])}
            fc = simulate_weekend(model, race, n=self.n_sims, seed=self.seed + t, known=known)
            tot = fc.total_points(include_dotd=p.dotd)
            pdnf = fc.dnf["race"].mean(0)
            for i, d in enumerate(fc.drivers):
                rows.append((t, d, float(tot[:, i].mean()), float(np.percentile(tot[:, i], 10)),
                             float(np.percentile(tot[:, i], 90)), float(pdnf[i])))
            ct = fc.constructor_total()
            cpit = fc.ccomp[("race", "pit")]
            known = ctruth[(ctruth["t"] == t)].set_index("team")["pit_known"]
            for j, team in enumerate(fc.teams):
                base = float((ct[:, j] - cpit[:, j]).mean())
                if bool(known.get(team, False)):
                    base += float(cpit[:, j].mean())
                crows.append((t, team, base))
        return Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred", "p10", "p90", "p_dnf"]),
                        pd.DataFrame(crows, columns=["t", "team", "pred"]))


class Baseline:
    kind = "baseline"

    def __init__(self, how="last3", alpha=0.3):
        self.how, self.alpha = how, alpha

    def describe(self):
        return {"how": self.how, "alpha": self.alpha}

    def predict(self, res, truth, ctruth, targets):
        rows = []
        for w in targets.itertuples():
            t = int(w.t)
            past = truth[truth["t"] < t]
            field_ = truth[truth["t"] == t][["driver"]]
            glob = past[past["t"] >= past["t"].max() - 2]["total"].mean()
            for d in field_["driver"]:
                h = past[past["driver"] == d].sort_values("t")["total"].values
                if self.how == "season":
                    hs = past[(past["driver"] == d) & (past["year"] == int(w.year))]["total"].values
                    h = hs if len(hs) else h[-5:]
                    val = h.mean() if len(h) else glob
                elif len(h) == 0:
                    val = glob
                elif self.how == "ewma":
                    wts = (1 - self.alpha) ** np.arange(len(h))[::-1]
                    val = float((h * wts).sum() / wts.sum())
                else:
                    k = int(self.how.replace("last", ""))
                    val = float(h[-k:].mean())
                rows.append((t, d, float(val)))
        return Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"]))


def _panel_features(truth):
    """Pre-weekend features per (weekend, driver); every feature uses strictly earlier weekends only."""
    df = truth.sort_values(["driver", "t"]).copy()
    g = df.groupby("driver", sort=False)
    df["pos_r"] = df["race_pos"].where(~df["race_dnf"], 22.0)
    df["dnf_i"] = df["race_dnf"].astype(float)

    def ew(s, a):
        return s.shift(1).ewm(alpha=a, adjust=False).mean()

    df["d_last1"] = g["total"].shift(1)
    df["d_last3"] = g["total"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["d_last5"] = g["total"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["d_ewm"] = g["total"].transform(lambda s: ew(s, 0.3))
    df["d_cnt"] = g.cumcount().clip(upper=30)
    df["d_qpos_ewm"] = g["quali_pos"].transform(lambda s: ew(s, 0.3))
    df["d_rpos_ewm"] = g["pos_r"].transform(lambda s: ew(s, 0.3))
    df["d_dnf_ewm"] = g["dnf_i"].transform(lambda s: ew(s, 0.15))
    # same-circuit history of the driver
    df["d_track_mean"] = df.groupby(["driver", "location"], sort=False)["total"].transform(
        lambda s: s.shift(1).expanding().mean())
    # team form: mean of the team's driver totals per weekend, then lagged
    tw = df.groupby(["team", "t"], as_index=False)["total"].mean().rename(columns={"total": "tm_total"})
    tw = tw.sort_values(["team", "t"])
    gt = tw.groupby("team", sort=False)
    tw["team_ewm"] = gt["tm_total"].transform(lambda s: s.shift(1).ewm(alpha=0.4, adjust=False).mean())
    tw["team_last3"] = gt["tm_total"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df = df.merge(tw[["team", "t", "team_ewm", "team_last3"]], on=["team", "t"], how="left")
    prof = {loc: track_profile(loc) for loc in df["location"].unique()}
    for k in ("stickiness", "chaos", "overtaking", "stops"):
        df[f"trk_{k}"] = df["location"].map(lambda l, k=k: prof[l][k])
    df["is_sprint"] = df["is_sprint"].astype(float)
    return df.sort_values(["t", "driver"]).reset_index(drop=True)


GBM_FEATURES = ["d_last1", "d_last3", "d_last5", "d_ewm", "d_cnt", "d_qpos_ewm", "d_rpos_ewm", "d_dnf_ewm",
                "d_track_mean", "team_ewm", "team_last3", "is_sprint",
                "trk_stickiness", "trk_chaos", "trk_overtaking", "trk_stops"]


class GBM:
    kind = "gbm"

    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5,
                 objective="reg:squarederror", min_train_rows=150, seed=0, extra=None, extra_cols=()):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                       min_child_weight=min_child_weight, objective=objective, subsample=0.8,
                       colsample_bytree=0.8, random_state=seed, n_jobs=1)
        self.min_rows = min_train_rows
        self.extra, self.extra_cols = extra, list(extra_cols)

    def describe(self):
        return {**self.kw, "min_train_rows": self.min_rows, "features": ",".join(GBM_FEATURES + self.extra_cols)}

    def predict(self, res, truth, ctruth, targets):
        from xgboost import XGBRegressor
        panel = _panel_features(truth)
        feats = GBM_FEATURES + self.extra_cols
        if self.extra is not None and self.extra_cols:
            panel = panel.merge(self.extra[["t", "driver"] + self.extra_cols], on=["t", "driver"], how="left")
        rows = []
        for w in targets.itertuples():
            t = int(w.t)
            tr = panel[panel["t"] < t].dropna(subset=["d_last1"])
            te = panel[panel["t"] == t]
            if len(tr) < self.min_rows:
                continue
            m = XGBRegressor(**self.kw).fit(tr[feats], tr["total"])
            pr = m.predict(te[feats])
            rows += [(t, d, float(v)) for d, v in zip(te["driver"], pr)]
        return Forecast(pd.DataFrame(rows, columns=["t", "driver", "pred"]))


def blend(a: Forecast, b: Forecast, w_a):
    """Convex combination of driver forecasts (rows present in both)."""
    m = a.drivers.merge(b.drivers[["t", "driver", "pred"]], on=["t", "driver"], suffixes=("", "_b"))
    out = m.copy()
    out["pred"] = w_a * m["pred"] + (1 - w_a) * m["pred_b"]
    for c in ("p10", "p90"):             # shift the sim interval with the blended centre
        if c in out:
            out[c] = out[c] + (out["pred"] - m["pred"])
    return Forecast(out.drop(columns=["pred_b"]), a.constructors)


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def _spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def per_weekend(fc: Forecast, truth, ctruth=None):
    d = fc.drivers.merge(truth[["t", "driver", "total", "race_dnf", "year", "round", "location"]],
                         on=["t", "driver"], how="inner")
    rows = []
    for t, g in d.groupby("t"):
        top_p = set(g.nlargest(5, "pred")["driver"])
        top_a = set(g.nlargest(5, "total")["driver"])
        r = {"t": t, "year": int(g["year"].iloc[0]), "round": int(g["round"].iloc[0]),
             "location": g["location"].iloc[0], "n": len(g),
             "mae": float((g["pred"] - g["total"]).abs().mean()),
             "bias": float((g["pred"] - g["total"]).mean()),
             "spearman": _spearman(g["pred"], g["total"]),
             "top5": len(top_p & top_a) / 5.0}
        if "p10" in g:
            r["cover80"] = float(((g["p10"] <= g["total"]) & (g["total"] <= g["p90"])).mean())
        if "p_dnf" in g:
            r["dnf_brier"] = float(((g["p_dnf"] - g["race_dnf"].astype(float)) ** 2).mean())
            r["p_dnf_mean"] = float(g["p_dnf"].mean())
            r["dnf_actual"] = float(g["race_dnf"].mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    if fc.constructors is not None and ctruth is not None and len(fc.constructors):
        c = fc.constructors.merge(ctruth[["t", "team", "total"]], on=["t", "team"])
        cm = c.assign(e=(c["pred"] - c["total"]).abs()).groupby("t")["e"].mean().rename("c_mae")
        out = out.merge(cm, on="t", how="left")
    return out


def summarize(pw):
    keys = ["mae", "spearman", "top5", "cover80", "dnf_brier", "p_dnf_mean", "dnf_actual", "c_mae", "bias"]
    s = {k: float(pw[k].mean()) for k in keys if k in pw}
    s["n_weekends"] = int(len(pw))
    return s


def paired_bootstrap(pw_a, pw_b, metric="mae", n=5000, seed=0):
    """a minus b over matched weekends (negative = a better for mae). Returns mean, 95% CI, P(a<b)."""
    m = pw_a[["t", metric]].merge(pw_b[["t", metric]], on="t", suffixes=("_a", "_b")).dropna()
    diff = (m[f"{metric}_a"] - m[f"{metric}_b"]).values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    bs = diff[idx].mean(1)
    return {"delta": float(diff.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p_a_lower": float((bs < 0).mean()), "n": int(len(diff))}


# ──────────────────────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────────────────────

def _mlflow():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def log_run(name, model, fc: Forecast, pw: pd.DataFrame, split, res, tags=None, extra_params=None,
            extra_metrics=None, parent=True):
    """Log one forecaster evaluated on one split."""
    import tempfile
    mlflow = _mlflow()
    with mlflow.start_run(run_name=name, nested=parent and mlflow.active_run() is not None):
        mlflow.set_tags({"split": split, "eval_year": SPLITS.get(split, ""), "model_kind": model.kind,
                         "git_sha": git_sha(), "data_fingerprint": data_fingerprint(res),
                         "protocol": PROTOCOL, "data_version": DATA_VERSION, **(tags or {})})
        params = {**model.describe(), **(extra_params or {})}
        mlflow.log_params({k: (str(v)[:240] if isinstance(v, str) else v) for k, v in params.items()})
        summ = summarize(pw)
        mlflow.log_metrics({**summ, **(extra_metrics or {})})
        for i, r in enumerate(pw.sort_values("t").itertuples()):
            step = int(r.round)
            m = {"wk_mae": r.mae, "wk_spearman": r.spearman, "wk_top5": r.top5}
            mlflow.log_metrics({k: float(v) for k, v in m.items() if pd.notna(v)}, step=step)
        with tempfile.TemporaryDirectory() as td:
            pw.to_csv(f"{td}/per_weekend.csv", index=False)
            fc.drivers.to_csv(f"{td}/driver_predictions.csv", index=False)
            mlflow.log_artifacts(td)
        return mlflow.active_run().info.run_id


def evaluate(model, split, hist=None, cur=None):
    res, truth, ctruth = get_data(hist, cur)
    targets = targets_for(res, SPLITS[split])
    t0 = time.time()
    fc = model.predict(res, truth, ctruth, targets)
    pw = per_weekend(fc, truth, ctruth)
    return fc, pw, time.time() - t0


def run_and_log(name, model, split, ref=None, tags=None, hist=None, cur=None, fc_override=None):
    """Evaluate `model` on `split`, compare with `ref` (= (name, per-weekend frame)), log to MLflow.
    Returns (forecast, per-weekend frame, summary)."""
    res, truth, ctruth = get_data(hist, cur)
    if fc_override is None:
        fc, pw, secs = evaluate(model, split, hist, cur)
    else:
        fc, secs = fc_override, 0.0
        pw = per_weekend(fc, truth, ctruth)
    summ = summarize(pw)
    extra = {"seconds": secs}
    if ref is not None:
        for metric in ("mae", "spearman"):
            if metric in pw and metric in ref[1]:
                bt = paired_bootstrap(pw, ref[1], metric)
                for k in ("delta", "lo", "hi", "p_a_lower"):
                    extra[f"vs_ref_{metric}_{k}"] = bt[k]
    tg = {"reference": ref[0] if ref else "none", **(tags or {})}
    log_run(name, model, fc, pw, split, res, tags=tg, extra_metrics=extra)
    return fc, pw, summ


if __name__ == "__main__":
    sys.exit("Import this module from the experiment scripts in experiments/ (see README).")
