#!/usr/bin/env python3
"""
Pre-lock practice features (f1_practice.py)
===========================================
Turns raw_practice_data.json (see fetch_practice.py) into per-(weekend, driver) features that are known
when F1 Fantasy teams lock (start of Qualifying; start of the Sprint on Sprint weekends):

  fp_gap_min / fp_gap_mean / fp_gap_last   best-lap gap to the session's fastest driver (%), across the sessions
  fp_rank_mean                             mean rank of the driver's best lap
  fp_lr_gap                                long-run pace gap (%): median of >=5 consistent laps in a stint
  fp_laps                                  clean laps run (participation)
  fp_team_gap_min / fp_vs_mate             team's best driver gap; driver minus teammate
  sq_pos / sq_rel                          Sprint Qualifying position (Sprint weekends only; known at lock)

Nothing here looks at Qualifying, the Sprint itself or the Race.
"""
import json
import os

import numpy as np
import pandas as pd

from f1_calendar import canon_driver, canon_location

PRACTICE_JSON = "raw_practice_data.json"
QUICK = 1.07            # a "representative" lap is within 107% of the session's fastest
FP_FEATURES = ["fp_gap_min", "fp_gap_mean", "fp_gap_last", "fp_rank_mean", "fp_lr_gap", "fp_laps",
               "fp_team_gap_min", "fp_vs_mate", "fp_n"]
SQ_FEATURES = ["sq_pos", "sq_rel"]


def load_practice(path=PRACTICE_JSON):
    return json.load(open(path)) if os.path.exists(path) else []


def _session_table(sess):
    """One practice session -> DataFrame(driver, best, lr, laps) with driver names."""
    names = {d["driver_number"]: canon_driver(d.get("last_name", "")) for d in sess["drivers"]}
    laps = pd.DataFrame(sess["laps"])
    if laps.empty:
        return pd.DataFrame(columns=["driver", "best", "lr", "laps"])
    laps = laps[laps["lap_duration"].notna() & (laps["lap_duration"] > 0) & ~laps["is_pit_out_lap"].fillna(False).astype(bool)]
    if laps.empty:
        return pd.DataFrame(columns=["driver", "best", "lr", "laps"])
    fastest = laps["lap_duration"].min()
    quick = laps[laps["lap_duration"] <= fastest * QUICK]
    out = quick.groupby("driver_number")["lap_duration"].agg(best="min", laps="size").reset_index()
    # long runs: laps inside a stint, trimmed of slow in/out/traffic laps, need >= 5 left
    lr = {}
    stints = pd.DataFrame(sess["stints"])
    if not stints.empty:
        for st in stints.itertuples():
            if pd.isna(st.lap_start) or pd.isna(st.lap_end):
                continue
            g = laps[(laps["driver_number"] == st.driver_number) & laps["lap_number"].between(st.lap_start, st.lap_end)]
            if len(g) < 5:
                continue
            med = g["lap_duration"].median()
            g = g[g["lap_duration"] <= med * 1.04]
            if len(g) >= 5:
                v = float(g["lap_duration"].median())
                lr[st.driver_number] = min(lr.get(st.driver_number, v), v)
    out["lr"] = out["driver_number"].map(lr)
    out["driver"] = out["driver_number"].map(names)
    return out.dropna(subset=["driver"])[["driver", "best", "lr", "laps"]]


def build_practice_features(pdata, weekends, sessions):
    """
    pdata     : load_practice()
    weekends  : Results.weekends (t, year, location)
    sessions  : Results.sessions (for Sprint Qualifying order and team membership)
    Returns DataFrame[t, driver, <FP_FEATURES>, <SQ_FEATURES>] for every driver in the weekend's race field.
    """
    key = {(int(r.year), r.location): int(r.t) for r in weekends.itertuples()}
    rows = []
    for m in pdata:
        t = key.get((int(m["year"]), canon_location(m["location"])))
        if t is None:
            continue
        per = []
        for s in m["sessions"]:
            tb = _session_table(s)
            if tb.empty:
                continue
            tb = tb.assign(gap=(tb["best"] / tb["best"].min() - 1) * 100,
                           rank=tb["best"].rank(method="min"),
                           lr_gap=(tb["lr"] / tb["lr"].min() - 1) * 100 if tb["lr"].notna().any() else np.nan)
            per.append(tb)
        if not per:
            continue
        allp = pd.concat([p.assign(idx=i) for i, p in enumerate(per)])
        g = allp.groupby("driver")
        f = pd.DataFrame({"fp_gap_min": g["gap"].min(), "fp_gap_mean": g["gap"].mean(),
                          "fp_gap_last": allp.sort_values("idx").groupby("driver")["gap"].last(),
                          "fp_rank_mean": g["rank"].mean(), "fp_lr_gap": g["lr_gap"].mean(),
                          "fp_laps": g["laps"].sum(), "fp_n": g.size()}).reset_index()
        f["t"] = t
        rows.append(f)
    fp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["t", "driver"] + FP_FEATURES)

    field = sessions[sessions["session"] == "R"][["t", "driver", "team"]]
    out = field.merge(fp, on=["t", "driver"], how="left")
    out["fp_team_gap_min"] = out.groupby(["t", "team"])["fp_gap_min"].transform("min")
    mate = out.groupby(["t", "team"])["fp_gap_min"].transform("sum") - out["fp_gap_min"].fillna(0)
    n_mates = out.groupby(["t", "team"])["fp_gap_min"].transform("count") - out["fp_gap_min"].notna().astype(int)
    out["fp_vs_mate"] = out["fp_gap_min"] - (mate / n_mates.replace(0, np.nan))
    sq = sessions[sessions["session"] == "SQ"][["t", "driver", "finish"]].rename(columns={"finish": "sq_pos"})
    sq["sq_rel"] = (sq["sq_pos"] - 1) / (sq.groupby("t")["sq_pos"].transform("max") - 1)
    out = out.merge(sq, on=["t", "driver"], how="left")
    return out[["t", "driver"] + FP_FEATURES + SQ_FEATURES]
