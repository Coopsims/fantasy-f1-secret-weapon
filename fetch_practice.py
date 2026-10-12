#!/usr/bin/env python3
"""
Practice-session cache (fetch_practice.py)
==========================================
Fetches what was known *before the F1 Fantasy team lock* for every weekend already in the raw caches:
  * lock time  = start of Qualifying, or of the Sprint on Sprint weekends (2026 rule, formula1.com previews;
                 confirmed against OpenF1 session start times)
  * Practice sessions that start before the lock (laps, stints, line-up) + the session schedule

Writes raw_practice_data.json:  [{meeting_key, year, location, lock_utc, sessions: [{...}]}]
Incomplete meetings (failed / rate-limited calls) are retried on the next run.
"""
import json
import os
import time

from f1_trainer import openf1_get
from openf1_utils import openf1_client

OUT = "raw_practice_data.json"
RAW = ("raw_historical_data.json", "raw_2026_results.json")
LAP_KEYS = ("driver_number", "lap_number", "lap_duration", "is_pit_out_lap", "date_start")


def lock_time(sessions):
    """Start of the Sprint if the weekend has one, else start of Qualifying (UTC ISO string)."""
    by = {s["session_name"]: s["date_start"] for s in sessions}
    return by.get("Sprint") or by.get("Qualifying")


def fetch_session(s):
    for _ in range(3):
        laps = openf1_get("laps", {"session_key": s["session_key"]})
        if laps:
            break
        time.sleep(1.0)
    stints = openf1_get("stints", {"session_key": s["session_key"]}) or []
    drivers = openf1_get("drivers", {"session_key": s["session_key"]}) or []
    return {"session_key": s["session_key"], "session_name": s["session_name"], "date_start": s["date_start"],
            "drivers": drivers, "stints": stints,
            "laps": [{k: l.get(k) for k in LAP_KEYS} for l in (laps or [])]}


def gaps(m):
    if not m.get("lock_utc"):
        return ["no lock time"]
    if not m["sessions"]:
        return ["no practice sessions"]
    return [f"{s['session_name']}: no laps/drivers" for s in m["sessions"] if not s["laps"] or not s["drivers"]]


def main():
    cache = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {m["meeting_key"] for m in cache if not gaps(m)}
    meetings = []
    for path in RAW:
        if os.path.exists(path):
            meetings += [(m["meeting_key"], m["year"], m["location"]) for m in json.load(open(path))
                         if "testing" not in (m.get("meeting_name") or "").lower()]
    openf1_client.check_refresh()
    for i, (mk, year, loc) in enumerate(meetings, 1):
        if mk in done:
            continue
        sessions = openf1_get("sessions", {"meeting_key": mk}) or []
        lock = lock_time(sessions)
        pre = [s for s in sessions if s["session_type"] == "Practice" and lock and s["date_start"] < lock]
        print(f"[{i}/{len(meetings)}] {year} {loc}: lock {lock}, {len(pre)} practice session(s) before it", flush=True)
        entry = {"meeting_key": mk, "year": year, "location": loc, "lock_utc": lock,
                 "schedule": [(s["session_name"], s["date_start"]) for s in sessions],
                 "sessions": [fetch_session(s) for s in sorted(pre, key=lambda s: s["date_start"])]}
        cache = [m for m in cache if m["meeting_key"] != mk] + [entry]
        json.dump(cache, open(OUT, "w"))
        if gaps(entry):
            print("   incomplete (will retry next run):", gaps(entry), flush=True)
        time.sleep(0.1)
    print("DONE", len(cache), "meetings;", sum(1 for m in cache if gaps(m)), "incomplete", flush=True)


if __name__ == "__main__":
    main()
