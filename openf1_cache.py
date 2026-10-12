#!/usr/bin/env python3
"""
Readers for the one-pull cache (openf1_cache.py)
================================================
Rebuilds the formats the rest of the project already understands from openf1_full/ (see fetch_openf1_full.py),
so nothing has to be downloaded twice:

    to_legacy_meetings(full)  -> same structure as raw_historical_data.json / raw_2026_results.json
    to_practice(full)         -> same structure as raw_practice_data.json (sessions that start before the team lock)
"""
import glob
import gzip
import json
import os
from collections import defaultdict

LAP_KEYS = ("driver_number", "lap_number", "lap_duration", "is_pit_out_lap", "date_start")
LEGACY_TYPES = ("Qualifying", "Race", "Sprint", "Sprint Qualifying")


def load_all(directory="openf1_full", only_complete=True):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            e = json.load(f)
        if e.get("sessions") and (e.get("complete") or not only_complete):
            out.append(e)
    return sorted(out, key=lambda e: e["meeting"]["date_start"])


def _by_session(rows):
    g = defaultdict(list)
    for r in rows or []:
        g[r["session_key"]].append(r)
    return g


def to_legacy_meetings(full):
    meetings = []
    for e in full:
        m, ep = e["meeting"], e["endpoints"]
        drivers, positions = _by_session(ep.get("drivers")), _by_session(ep.get("position"))
        laps, stints, pit = _by_session(ep.get("laps")), _by_session(ep.get("stints")), _by_session(ep.get("pit"))
        sessions = []
        for s in sorted(e["sessions"], key=lambda s: s["date_start"]):
            if s["session_type"] not in LEGACY_TYPES:
                continue
            sk, race_like = s["session_key"], s["session_type"] in ("Race", "Sprint")
            sessions.append({"session_key": sk, "session_type": s["session_type"], "session_name": s["session_name"],
                             "drivers": drivers.get(sk, []), "positions": positions.get(sk, []),
                             "laps": laps.get(sk, []) if race_like else [],
                             "stints": stints.get(sk, []) if race_like else [],
                             "pit": pit.get(sk, []) if race_like else []})
        first = sessions[0]["drivers"] if sessions else []
        meetings.append({"year": m["year"], "location": m["location"], "meeting_key": m["meeting_key"],
                         "meeting_name": m.get("meeting_name", ""), "date_start": m["date_start"],
                         "drivers": first, "sessions": sessions})
    return meetings


def lock_time(sessions):
    """2026 F1 Fantasy rule: start of the Sprint if there is one, else start of Qualifying (formula1.com previews)."""
    by = {s["session_name"]: s["date_start"] for s in sessions}
    return by.get("Sprint") or by.get("Qualifying")


def to_practice(full):
    out = []
    for e in full:
        m, ep = e["meeting"], e["endpoints"]
        lock = lock_time(e["sessions"])
        drivers, laps, stints = _by_session(ep.get("drivers")), _by_session(ep.get("laps")), _by_session(ep.get("stints"))
        pre = [s for s in e["sessions"] if s["session_type"] == "Practice" and lock and s["date_start"] < lock]
        out.append({"meeting_key": m["meeting_key"], "year": m["year"], "location": m["location"], "lock_utc": lock,
                    "schedule": [(s["session_name"], s["date_start"]) for s in e["sessions"]],
                    "sessions": [{"session_key": s["session_key"], "session_name": s["session_name"],
                                  "date_start": s["date_start"], "drivers": drivers.get(s["session_key"], []),
                                  "stints": stints.get(s["session_key"], []),
                                  "laps": [{k: l.get(k) for k in LAP_KEYS} for l in laps.get(s["session_key"], [])]}
                                 for s in sorted(pre, key=lambda s: s["date_start"])]})
    return out
