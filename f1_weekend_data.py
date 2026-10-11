#!/usr/bin/env python3
"""
Tidy per-session race data (f1_weekend_data.py)
===============================================
Turns the raw OpenF1 caches that the trainer already writes
(raw_historical_data.json / raw_2026_results.json) - or, failing that, the
training CSVs - into two clean tables the weekend model works from:

  sessions : one row per driver per session
             (year, round, t, location, session in {Q, SQ, S, R}, driver, team,
              grid, finish, dnf, fastest_lap, overtakes, n_cars)
  pits     : one row per pit stop with a *stationary* time where available
  weekends : one row per race weekend (t, year, round, location, sprint)

It also scores real results with the same rules module the simulator uses, so
"actual" and "predicted" points are always computed identically.

No network access and no sklearn / xgboost imports.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import f1_rules as R
from f1_calendar import (
    canon_driver, canon_team, canon_location, historical_round, round_for_location,
)

HISTORICAL_JSON = "raw_historical_data.json"
CURRENT_JSON = "raw_2026_results.json"
DRIVER_CSV = "f1_driver_training_data.csv"

SESSION_ORDER = ["Q", "SQ", "S", "R"]


@dataclass
class Results:
    sessions: pd.DataFrame
    pits: pd.DataFrame
    weekends: pd.DataFrame
    truth: object = None      # only set by the synthetic generator

    @property
    def empty(self):
        return self.sessions is None or self.sessions.empty


# ──────────────────────────────────────────────────────────────
# Raw-session helpers
# ──────────────────────────────────────────────────────────────

def _latest_positions(positions_list):
    """Latest position record per driver number (records are chronological)."""
    latest = {}
    for p in positions_list or []:
        dn, pos = p.get("driver_number"), p.get("position")
        if dn is not None and pos is not None:
            latest[dn] = pos
    return latest


def _find_fastest_lap(laps_list):
    best, who = None, None
    for lap in laps_list or []:
        dur, dn = lap.get("lap_duration"), lap.get("driver_number")
        if dur and dn and (best is None or dur < best):
            best, who = dur, dn
    return who


def _detect_dnf(stints_list, laps_list):
    """Driver numbers that completed < 90% of the distance of the longest runner."""
    max_lap = {}
    for s in stints_list or []:
        dn, end = s.get("driver_number"), s.get("lap_end") or 0
        if dn:
            max_lap[dn] = max(max_lap.get(dn, 0), end)
    if not max_lap:  # fall back to lap records
        for lap in laps_list or []:
            dn, ln = lap.get("driver_number"), lap.get("lap_number") or 0
            if dn:
                max_lap[dn] = max(max_lap.get(dn, 0), ln)
    if not max_lap:
        return set()
    overall = max(max_lap.values())
    return {dn for dn, laps in max_lap.items() if laps < overall * 0.9}


def compute_overtakes(positions, laps, pits):
    """
    On-track passes per driver number, from the position stream.

    The legacy trainer read a `position` field off the laps endpoint, which
    OpenF1 does not provide, so overtakes were silently always 0.  Here the
    order of the field is sampled on a common clock at the end of every lap and
    a pass is counted whenever driver d was behind car e one lap earlier and is
    ahead now.  Pairs involving a car that pitted (or was on an out-lap) during
    the lap, or that has already retired, are ignored so that pit-cycle shuffles
    aren't counted as overtakes.

    Returns {driver_number: passes} or None when the data can't support it.
    """
    if not positions or not laps:
        return None
    try:
        pos = pd.DataFrame(positions)[["driver_number", "position", "date"]].dropna()
        pos["date"] = pd.to_datetime(pos["date"], utc=True, errors="coerce")
        pos = pos.dropna().sort_values("date")
        lp = pd.DataFrame(laps)
        lp["date_start"] = pd.to_datetime(lp["date_start"], utc=True, errors="coerce")
        lp = lp.dropna(subset=["date_start", "lap_number", "driver_number"])
    except (KeyError, ValueError):
        return None
    if pos.empty or lp.empty:
        return None

    drivers = sorted(set(pos["driver_number"]))
    didx = {d: i for i, d in enumerate(drivers)}
    n = len(drivers)
    n_laps = int(lp["lap_number"].max())
    if n_laps < 2:
        return None

    # Common clock: the (median) time a lap starts, i.e. the previous lap ends.
    start_t = lp.groupby("lap_number")["date_start"].median()
    t0 = start_t.get(1, pos["date"].min())
    clock = [t0] + [start_t.get(L + 1, None) for L in range(1, n_laps)]
    clock = [c for c in clock]
    # Cars' order at each clock tick = last position record at or before it.
    per_driver = {d: g for d, g in pos.groupby("driver_number")}

    def order_at(ts):
        out = np.full(n, np.nan)
        for d, g in per_driver.items():
            k = g["date"].searchsorted(ts, side="right") - 1
            if k >= 0:
                out[didx[d]] = g["position"].iloc[k]
        return out

    # Pit involvement per (lap, driver)
    in_pit = {}
    for p in pits or []:
        dn, ln = p.get("driver_number"), p.get("lap_number")
        if dn in didx and ln:
            in_pit.setdefault(int(ln), set()).add(dn)
    if "is_pit_out_lap" in lp.columns:
        for _, row in lp[lp["is_pit_out_lap"] == True].iterrows():  # noqa: E712
            in_pit.setdefault(int(row["lap_number"]), set()).add(row["driver_number"])
    last_lap = lp.groupby("driver_number")["lap_number"].max().to_dict()

    counts = np.zeros(n)
    prev = order_at(clock[0])
    for L in range(1, n_laps + 1):
        ts = clock[L] if L < len(clock) else None
        if ts is None or pd.isna(ts):
            break
        cur = order_at(ts)
        busy = np.zeros(n, dtype=bool)
        for dn in in_pit.get(L, set()) | in_pit.get(L - 1, set()):
            busy[didx[dn]] = True
        for d in drivers:  # retired before this lap ended
            if last_lap.get(d, n_laps) < L:
                busy[didx[d]] = True
        ok = ~busy & ~np.isnan(prev) & ~np.isnan(cur)
        ahead_before = prev[None, :] < prev[:, None]       # [d, e]: e ahead of d
        behind_now = cur[None, :] > cur[:, None]            # [d, e]: e behind d now
        passed = ahead_before & behind_now & ok[None, :] & ok[:, None]
        counts += passed.sum(axis=1)
        prev = cur

    result = {d: int(counts[didx[d]]) for d in drivers}
    if result and np.mean(list(result.values())) > 30:   # garbage in, garbage out
        return None
    return result


def _relabel_sessions(raw_sessions):
    """
    OpenF1 labels sprint sessions as plain 'Qualifying' / 'Race'.  On a sprint
    weekend there are two of each; the lower session_key of each pair is the
    sprint variant.  Honours an explicit session_name when the cache has one.
    """
    sessions = sorted(raw_sessions, key=lambda s: s["session_key"])
    labelled = []
    q = [s for s in sessions if s.get("session_type") == "Qualifying"]
    r = [s for s in sessions if s.get("session_type") == "Race"]
    sprint_q = {id(q[0])} if len(q) == 2 and len(r) == 2 else set()
    sprint_r = {id(r[0])} if len(q) == 2 and len(r) == 2 else set()
    for s in sessions:
        stype = s.get("session_type")
        name = (s.get("session_name") or "").lower()
        if stype in ("Sprint", "Sprint Qualifying") or "sprint" in name:
            code = "SQ" if ("qual" in name or "shootout" in name or stype == "Sprint Qualifying") else "S"
        elif stype == "Qualifying":
            code = "SQ" if id(s) in sprint_q else "Q"
        elif stype == "Race":
            code = "S" if id(s) in sprint_r else "R"
        else:
            continue
        labelled.append((code, s))
    return labelled


# ──────────────────────────────────────────────────────────────
# Build tidy tables
# ──────────────────────────────────────────────────────────────

def _meeting_rows(m):
    """Convert one raw meeting dict into (session_rows, pit_rows) or None."""
    drivers = m.get("drivers", [])
    dmap = {d["driver_number"]: canon_driver(d.get("last_name", "")) for d in drivers}
    tmap = {d["driver_number"]: canon_team(d.get("team_name", "Unknown")) for d in drivers}
    labelled = _relabel_sessions(m.get("sessions", []))

    by_code = {}
    for code, s in labelled:
        by_code[code] = s
    if "Q" not in by_code and "R" not in by_code:
        return None

    rows, pit_rows = [], []
    quali_pos = {}   # name -> position, to attach grids
    sq_pos = {}

    for code in ["Q", "SQ", "R", "S"]:
        s = by_code.get(code)
        if s is None:
            continue
        latest = _latest_positions(s.get("positions"))
        finish = {dmap[dn]: int(p) for dn, p in latest.items() if dn in dmap and dmap[dn]}
        if not finish:
            continue
        if code == "Q":
            quali_pos = finish
        if code == "SQ":
            sq_pos = finish

        dnf_nums, fl_num, ot = set(), None, {}
        if code in ("R", "S"):
            dnf_nums = _detect_dnf(s.get("stints"), s.get("laps"))
            fl_num = _find_fastest_lap(s.get("laps"))
            ot_raw = compute_overtakes(s.get("positions"), s.get("laps"), s.get("pit"))
            ot = {dmap[dn]: v for dn, v in (ot_raw or {}).items() if dn in dmap}
            if code == "R":
                for p in s.get("pit") or []:
                    dn = p.get("driver_number")
                    stop = p.get("stop_duration")
                    if dn in dmap and isinstance(stop, (int, float)) and 1.0 <= stop <= 12.0:
                        pit_rows.append({
                            "team": tmap.get(dn, "Unknown"), "driver": dmap[dn],
                            "lap": p.get("lap_number"), "stop_s": float(stop),
                        })
        grid_src = quali_pos if code == "R" else (sq_pos if code == "S" else {})
        for dn in latest:
            name = dmap.get(dn)
            if not name or name not in finish:
                continue
            rows.append({
                "session": code, "driver": name, "team": tmap.get(dn, "Unknown"),
                "grid": grid_src.get(name, np.nan),
                "finish": finish[name],
                "dnf": dn in dnf_nums,
                "fastest_lap": fl_num is not None and dn == fl_num,
                "overtakes": ot.get(name, np.nan),
                "n_cars": len(finish),
            })
    return rows, pit_rows


def build_from_meetings(meetings):
    """Raw OpenF1 meeting dicts -> Results."""
    seen, ordered = set(), []
    for m in sorted(meetings, key=lambda m: (m.get("year", 0), m.get("meeting_key", 0))):
        key = m.get("meeting_key")
        if key in seen:
            continue
        seen.add(key)
        ordered.append(m)

    all_rows, all_pits, wk_rows = [], [], []
    order_in_year = {}
    for m in ordered:
        built = _meeting_rows(m)
        if built is None:
            continue
        rows, pit_rows = built
        if not rows:
            continue
        year = m["year"]
        order_in_year[year] = order_in_year.get(year, 0) + 1
        rnd = historical_round(year, m["location"]) or order_in_year[year]
        loc = canon_location(m["location"])
        is_sprint = any(r["session"] in ("S", "SQ") for r in rows)
        wk_rows.append({"year": year, "round": rnd, "order": order_in_year[year],
                        "location": loc, "sprint": is_sprint})
        for r in rows:
            r.update({"year": year, "round": rnd, "location": loc})
        for p in pit_rows:
            p.update({"year": year, "round": rnd, "location": loc})
        all_rows.extend(rows)
        all_pits.extend(pit_rows)

    return _finalise(all_rows, all_pits, wk_rows)


def build_from_training_csv(path=DRIVER_CSV):
    """
    Fallback when only the processed training CSV exists.  It has no grid
    penalties, pit stops or sprint qualifying, but carries qualifying / race /
    sprint positions, DNFs and fastest laps - enough for the core model.
    """
    df = pd.read_csv(path)
    rows, wk_rows, seen = [], [], {}
    for (year, loc), g in df.groupby(["year", "location"], sort=False):
        seen[year] = seen.get(year, 0) + 1
        rnd = historical_round(year, loc) or seen[year]
        loc_c = canon_location(loc)
        sprint = bool(g.get("is_sprint_weekend", pd.Series([0])).max())
        wk_rows.append({"year": int(year), "round": rnd, "order": seen[year],
                        "location": loc_c, "sprint": sprint})
        n_q = int(g["quali_pos"].notna().sum())
        n_r = int(g["race_pos"].notna().sum() + g["dnf"].sum())
        for _, r in g.iterrows():
            base = {"year": int(year), "round": rnd, "location": loc_c,
                    "driver": canon_driver(r["driver"]), "team": canon_team(r["team"])}
            if pd.notna(r.get("quali_pos")):
                rows.append({**base, "session": "Q", "grid": np.nan, "finish": int(r["quali_pos"]),
                             "dnf": False, "fastest_lap": False, "overtakes": np.nan, "n_cars": n_q})
            if pd.notna(r.get("race_pos")) or r.get("dnf", 0) == 1:
                rows.append({**base, "session": "R",
                             "grid": r["quali_pos"] if pd.notna(r.get("quali_pos")) else np.nan,
                             "finish": int(r["race_pos"]) if pd.notna(r.get("race_pos")) else 99,
                             "dnf": bool(r.get("dnf", 0) == 1),
                             "fastest_lap": bool(r.get("fastest_lap", 0) == 1),
                             "overtakes": r.get("overtakes", np.nan), "n_cars": n_r})
            if pd.notna(r.get("sprint_pos")):
                rows.append({**base, "session": "S", "grid": np.nan, "finish": int(r["sprint_pos"]),
                             "dnf": False, "fastest_lap": False, "overtakes": np.nan,
                             "n_cars": int(g["sprint_pos"].notna().sum())})
    return _finalise(rows, [], wk_rows)


def _finalise(rows, pit_rows, wk_rows):
    sessions = pd.DataFrame(rows)
    pits = pd.DataFrame(pit_rows)
    weekends = pd.DataFrame(wk_rows)
    if weekends.empty:
        return Results(pd.DataFrame(), pits, weekends)
    weekends = weekends.sort_values(["year", "order"]).reset_index(drop=True)
    weekends["t"] = np.arange(len(weekends))
    tmap = {(r.year, r.round): r.t for r in weekends.itertuples()}
    sessions["t"] = [tmap[(y, r)] for y, r in zip(sessions["year"], sessions["round"])]
    if not pits.empty:
        pits["t"] = [tmap[(y, r)] for y, r in zip(pits["year"], pits["round"])]
    sessions["dnf"] = sessions["dnf"].astype(bool)
    sessions["fastest_lap"] = sessions["fastest_lap"].astype(bool)
    return Results(sessions.sort_values(["t", "session", "finish"]).reset_index(drop=True),
                   pits, weekends)


def load_results(historical=HISTORICAL_JSON, current=CURRENT_JSON, csv=DRIVER_CSV,
                 verbose=True):
    """
    Load whatever race data exists locally.  Preference order:
      raw JSON caches (full detail)  ->  training CSV (+ any 2026 JSON).
    Returns an (possibly empty) Results.
    """
    meetings = []
    for path in (historical, current):
        if path and os.path.exists(path):
            with open(path, "r") as f:
                meetings.extend(json.load(f))
    if meetings:
        res = build_from_meetings(meetings)
        if not res.empty:
            if verbose:
                print(f"  📂 Loaded {len(res.weekends)} race weekends from raw caches")
            return res
    if csv and os.path.exists(csv):
        res = build_from_training_csv(csv)
        if current and os.path.exists(current):  # layer 2026 results over the CSV history
            with open(current, "r") as f:
                cur = build_from_meetings(json.load(f))
            res = _merge(res, cur)
        if verbose and not res.empty:
            print(f"  📂 Loaded {len(res.weekends)} race weekends from training CSV")
        return res
    return Results(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def _merge(a, b):
    """Concatenate two Results (b's years override a's)."""
    if b.empty:
        return a
    if a.empty:
        return b
    years_b = set(b.weekends["year"])
    keep_w = a.weekends[~a.weekends["year"].isin(years_b)]
    keep_s = a.sessions[~a.sessions["year"].isin(years_b)]
    keep_p = a.pits[~a.pits["year"].isin(years_b)] if not a.pits.empty else a.pits
    rows = keep_s.drop(columns=["t"]).to_dict("records") + b.sessions.drop(columns=["t"]).to_dict("records")
    pits = (keep_p.drop(columns=["t"], errors="ignore").to_dict("records")
            + b.pits.drop(columns=["t"], errors="ignore").to_dict("records"))
    wk = keep_w.drop(columns=["t"]).to_dict("records") + b.weekends.drop(columns=["t"]).to_dict("records")
    return _finalise(rows, pits, wk)


# ──────────────────────────────────────────────────────────────
# Score real results with the shared rules
# ──────────────────────────────────────────────────────────────

def score_weekend_actual(res, t):
    """
    Fantasy points actually earned at weekend index `t`.

    Returns (driver_df, constructor_df).
      driver_df       : driver, team, quali, sprint, race, total  (+ per-component cols)
      constructor_df  : team, quali, sprint, race, total, pit_known

    Driver of the Day is not in OpenF1, and pit points are only included when
    stationary stop times exist, so `pit_known` flags whether the constructor
    total includes them.
    """
    ss = res.sessions[res.sessions["t"] == t]
    if ss.empty:
        return pd.DataFrame(), pd.DataFrame()

    drivers = {}
    def slot(name, team):
        return drivers.setdefault(name, {"driver": name, "team": team, "quali": 0.0,
                                         "sprint": 0.0, "race": 0.0})

    for _, row in ss[ss["session"] == "Q"].iterrows():
        d = slot(row["driver"], row["team"])
        d["quali"] = float(R.score_qualifying(row["finish"]))
        d["quali_pos"] = int(row["finish"])

    for code, kind, key in (("R", "race", "race"), ("S", "sprint", "sprint")):
        g = ss[ss["session"] == code]
        for _, row in g.iterrows():
            d = slot(row["driver"], row["team"])
            grid = row["grid"] if pd.notna(row["grid"]) else row["finish"]
            comp = R.score_race_like(
                kind, grid, row["finish"], row["dnf"],
                0 if pd.isna(row["overtakes"]) else row["overtakes"], row["fastest_lap"])
            d[key] = float(R.event_total(comp))
            for k, v in comp.items():
                d[f"{key}_{k}"] = float(v)
            d[f"{key}_pos"] = None if row["dnf"] else int(row["finish"])
            d[f"{key}_dnf"] = bool(row["dnf"])

    ddf = pd.DataFrame(list(drivers.values())).fillna(0)
    ddf["total"] = ddf[["quali", "sprint", "race"]].sum(axis=1)

    cons = []
    pits = res.pits[res.pits["t"] == t] if res.pits is not None and not res.pits.empty else None
    for team, g in ddf.groupby("team"):
        q = g["quali"].sum()
        qpos = [p for p in g.get("quali_pos", pd.Series(dtype=float)) if p]
        if len(qpos) == len(g) and len(qpos) == 2:
            q += float(R.constructor_quali_bonus(qpos[0], qpos[1]))
        elif len(qpos) == 1:
            q += float(R.constructor_quali_bonus(qpos[0], 99))
        entry = {"team": team, "quali": q, "sprint": g["sprint"].sum(),
                 "race": g["race"].sum(), "pit_known": False}
        if pits is not None and not pits.empty:
            tp = pits[pits["team"] == team]
            if not tp.empty:
                entry["race"] += float(R.pit_stop_points(tp["stop_s"].values).sum())
                entry["pit_known"] = True
        entry["total"] = entry["quali"] + entry["sprint"] + entry["race"]
        cons.append(entry)
    cdf = pd.DataFrame(cons)
    if pits is not None and not pits.empty and cdf["pit_known"].any():
        fastest = pits.loc[pits["stop_s"].idxmin(), "team"]
        m = cdf["team"] == fastest
        cdf.loc[m, "race"] += R.PIT_STOP_FASTEST_BONUS
        cdf.loc[m, "total"] += R.PIT_STOP_FASTEST_BONUS
    return ddf, cdf
