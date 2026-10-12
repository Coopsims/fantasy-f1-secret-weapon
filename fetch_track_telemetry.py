#!/usr/bin/env python3
"""
Track telemetry (fetch_track_telemetry.py)
==========================================
For every weekend in the one-pull cache, fetches OpenF1 car_data (speed, throttle, brake, gear, DRS at ~3.8 Hz)
for the 3 fastest PRACTICE laps set before the team lock (by different drivers) - one small date-filtered
request per lap, paced at 1.1 s.  Practice laps keep the descriptors free of qualifying / race information.

    python fetch_track_telemetry.py          # -> openf1_full/telemetry/<year>_<meeting_key>.json.gz (resumable)
"""
import gzip
import json
import os

import pandas as pd

import openf1_cache as C
from openf1_utils import OpenF1Error, openf1_client

OUT = os.path.join("openf1_full", "telemetry")


def fastest_practice_laps(e, n=3):
    lock = C.lock_time(e["sessions"])
    pre = {s["session_key"] for s in e["sessions"] if s["session_name"].startswith("Practice") and lock and s["date_start"] < lock}
    laps = pd.DataFrame(e["endpoints"].get("laps") or [])
    if laps.empty:
        return []
    laps = laps[laps["session_key"].isin(pre) & laps["lap_duration"].notna() & laps["date_start"].notna()
                & ~laps["is_pit_out_lap"].fillna(False).astype(bool)]
    best = laps.sort_values("lap_duration").drop_duplicates("driver_number").head(n)
    return best[["session_key", "driver_number", "lap_number", "lap_duration", "date_start"]].to_dict("records")


def main():
    os.makedirs(OUT, exist_ok=True)
    openf1_client.check_refresh()
    openf1_client.set_pacing(1.1)
    full = C.load_all()
    for i, e in enumerate(full, 1):
        m = e["meeting"]
        path = os.path.join(OUT, f"{m['year']}_{m['meeting_key']}.json.gz")
        if os.path.exists(path):
            continue
        out = {"meeting": m, "laps": []}
        for lap in fastest_practice_laps(e):
            start = pd.Timestamp(lap["date_start"])
            end = start + pd.Timedelta(seconds=float(lap["lap_duration"]))
            q = (f"car_data?session_key={lap['session_key']}&driver_number={lap['driver_number']}"
                 f"&date>={start.isoformat()}&date<={end.isoformat()}")
            try:
                rows = openf1_client.get_strict(q)
                status = "ok" if rows else "empty"
            except OpenF1Error as err:
                rows, status = [], f"error: {err}"
            out["laps"].append({**lap, "status": status,
                                "samples": [{k: r.get(k) for k in ("date", "speed", "throttle", "brake", "n_gear", "rpm", "drs")} for r in rows]})
        if all(l["status"] != "ok" for l in out["laps"]):
            print(f"[{i}/{len(full)}] {m['year']} {m['location']}: no telemetry ({[l['status'] for l in out['laps']]}) - not saved")
            continue
        with gzip.open(path, "wt") as f:
            json.dump(out, f)
        print(f"[{i}/{len(full)}] {m['year']} {m['location']}: {[len(l['samples']) for l in out['laps']]} samples", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
