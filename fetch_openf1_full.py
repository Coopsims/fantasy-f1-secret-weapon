#!/usr/bin/env python3
"""
One-pull OpenF1 cache (fetch_openf1_full.py)
============================================
Pulls everything useful for a race weekend ONCE, at meeting level (one request per endpoint per weekend instead of
one per session per endpoint - verified to return exactly the same rows), and stores it in openf1_full/.
Later features (practice pace, weather, race control, official overtakes, ...) are then derived locally; nothing
needs to be re-downloaded.  See docs/OPENF1_NOTES.md for rate-limit notes and the data layout.

    python fetch_openf1_full.py                      # 2023-2026, all light endpoints + intervals
    python fetch_openf1_full.py --years 2026         # only some seasons
    python fetch_openf1_full.py --no-intervals       # skip the biggest endpoint (~3 MB / race weekend)
    python fetch_openf1_full.py --pace 1.5           # seconds between requests (default 1.1)

Each file openf1_full/<year>_<meeting_key>.json.gz holds
    {"meeting": {...}, "sessions": [...], "complete": bool, "fetched_at": iso,
     "endpoints": {name: [rows]}, "status": {name: "ok" | "empty" | "error: ..."}}
"ok" = rows returned, "empty" = OpenF1 answered with no rows (can be genuine, e.g. no overtakes data in old seasons),
"error" = the request failed; errors (and weekends still in progress) are retried on the next run, nothing else is.
Deliberately NOT pulled: car_data and location (per-sample telemetry, orders of magnitude larger; not measured here).
"""
import argparse
import datetime as dt
import gzip
import json
import os
import time

from openf1_utils import OpenF1Error, openf1_client

OUT_DIR = "openf1_full"
LIGHT = ["drivers", "laps", "position", "stints", "pit", "weather", "race_control", "session_result",
         "starting_grid", "overtakes", "team_radio", "championship_drivers", "championship_teams"]
HEAVY = ["intervals"]            # ~3 MB per race weekend (race sessions only)


def path_for(year, mk):
    return os.path.join(OUT_DIR, f"{year}_{mk}.json.gz")


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def save(path, obj):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def pull(endpoint, **params):
    """(rows, status) - never raises, never mistakes a failure for 'no data'."""
    try:
        rows = openf1_client.get_strict(endpoint, params)
    except OpenF1Error as e:
        return [], f"error: {str(e)[:160]}"
    return rows, ("ok" if rows else "empty")


def weekend_complete(sessions, now):
    ends = [s["date_end"] for s in sessions if s.get("date_end")]
    if not ends:
        return False
    return dt.datetime.fromisoformat(max(ends)) + dt.timedelta(hours=3) < now


def fetch_meeting(meeting, endpoints, now):
    year, mk = meeting["year"], meeting["meeting_key"]
    path = path_for(year, mk)
    entry = load(path) if os.path.exists(path) else {"meeting": meeting, "sessions": [], "endpoints": {}, "status": {}}
    todo = [e for e in endpoints if entry["status"].get(e, "error").startswith("error")
            or not entry.get("complete", False)]
    if entry["sessions"] and entry.get("complete") and not todo and entry["status"].get("sessions") != "error":
        return "cached", entry
    sessions, st = pull("sessions", meeting_key=mk)
    entry["status"]["sessions"] = st
    if st.startswith("error"):
        save(path, entry)
        return "sessions failed", entry
    entry["sessions"] = sessions
    entry["meeting"] = meeting
    entry["complete"] = weekend_complete(sessions, now)
    if not entry["complete"]:
        todo = list(endpoints)                      # weekend still running: take everything again next time
    for ep in todo:
        rows, st = pull(ep, meeting_key=mk)
        entry["endpoints"][ep], entry["status"][ep] = rows, st
    entry["fetched_at"] = now.isoformat(timespec="seconds")
    save(path, entry)
    return "fetched", entry


def meetings_for(years, now):
    out = []
    for y in years:
        rows = openf1_client.get_strict("meetings", {"year": y})
        for m in rows:
            if "testing" in (m.get("meeting_name") or "").lower() or m.get("is_cancelled"):
                continue
            if dt.datetime.fromisoformat(m["date_start"]) > now:
                continue                              # not started yet
            out.append(m)
    return sorted(out, key=lambda m: m["date_start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--no-intervals", action="store_true")
    ap.add_argument("--pace", type=float, default=1.1, help="minimum seconds between requests; 1.1 ran 81 weekends with no 429, 0.35 hit one (raised automatically after a 429)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    openf1_client.check_refresh()
    openf1_client.set_pacing(a.pace)
    endpoints = LIGHT + ([] if a.no_intervals else HEAVY)
    now = dt.datetime.now(dt.timezone.utc)
    meetings = meetings_for(a.years, now)
    print(f"{len(meetings)} weekends, {len(endpoints)} endpoints each, pacing {a.pace}s", flush=True)
    bad = []
    for i, m in enumerate(meetings, 1):
        what, entry = fetch_meeting(m, endpoints, now)
        errs = [k for k, v in entry["status"].items() if v.startswith("error")]
        print(f"[{i}/{len(meetings)}] {m['year']} {m['location']:<18s} {what:<14s} "
              f"{'complete' if entry.get('complete') else 'IN PROGRESS':<11s} errors: {errs or '-'}", flush=True)
        if errs or not entry.get("complete"):
            bad.append((m["year"], m["location"], errs))
    print(f"DONE. {len(meetings) - len(bad)} clean weekends; {len(bad)} with errors or still running: {bad}", flush=True)


if __name__ == "__main__":
    main()
