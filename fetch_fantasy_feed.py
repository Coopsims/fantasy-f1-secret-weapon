#!/usr/bin/env python3
"""
Official F1 Fantasy game feed (fetch_fantasy_feed.py)
=====================================================
The game (fantasy.formula1.com) serves public JSON feeds. Per gameday (= raced weekend, numbered 1, 2, 3...
consecutively; cancelled races are skipped, so Miami 2026 is gameday 4) the drivers feed lists all 22 drivers
and 11 constructors with:
    Value / OldPlayerValue      price for that gameday / before it
    GamedayPoints               official fantasy points of that gameday, incl. Driver of the Day and pit stops
    SessionWisePoints           per session (AdditionalStats are season-to-date and inconsistent between files: unused)
This fills the two gaps OpenF1 cannot: the game's own points and the real price of every round.

    python fetch_fantasy_feed.py            # completed gamedays + the next one (current prices)

Writes (raw JSON kept in fantasy_feed/, gitignored):
    fantasy_f1_official_points.csv   round,name,points[,dotd,...]  for  f1_season.py --official-points
    fantasy_f1_prices_official.csv   round,date,name,type,price    for  f1_season.py --prices
`round` is this project's calendar round (f1_calendar.RACE_CALENDAR_2026), mapped by race order, so the
files line up with the rest of the pipeline; the game's gameday number is kept in a `gameday` column.
Requests are paced (1.5 s) and each file is fetched once; re-runs only fetch gamedays not saved yet
(plus the current one, whose prices may still change).
"""
import json
import os
import time

import pandas as pd
import requests

from f1_calendar import GRID_2026, RACE_CALENDAR_2026, canon_driver, canon_team

BASE = "https://fantasy.formula1.com/feeds"
OUT = "fantasy_feed"
UA = {"User-Agent": "Mozilla/5.0 (fantasy-f1-secret-weapon; personal analysis)"}
PACE = 1.5


def get(path, refresh=False):
    os.makedirs(OUT, exist_ok=True)
    local = os.path.join(OUT, path.replace("/", "_"))
    if os.path.exists(local) and not refresh:
        return json.load(open(local))
    for attempt in range(4):
        r = requests.get(f"{BASE}/{path}", headers=UA, timeout=30)
        time.sleep(PACE)
        if r.status_code == 200:
            data = r.json()
            json.dump(data, open(local, "w"))
            return data
        if r.status_code in (403, 404):
            return None
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{path}: HTTP {r.status_code}")


def gamedays():
    """[(gameday, meeting_name, locked, first_session_start)] from the schedule feed."""
    s = get("schedule/raceday_en.json", refresh=True)["Data"]["Value"]
    out = {}
    for e in s:
        g = int(e["GamedayId"])
        out.setdefault(g, (g, e["MeetingName"], int(e["GDIsLocked"]) == 1, e["GameDate"]))
    return [out[g] for g in sorted(out)]


def name_of(p):
    if p["PositionName"].upper().startswith("DRIVER"):
        return canon_driver(p.get("LastName") or p["FUllName"].split()[-1])
    return canon_team(p.get("TeamName") or p["FUllName"])


def main():
    gds = gamedays()
    done = [g for g in gds if g[2]]
    nxt = next((g for g in gds if not g[2]), None)
    raced_rounds = [r["round"] for r in RACE_CALENDAR_2026 if not r.get("cancelled")]
    print(f"{len(done)} completed gamedays; next: {nxt[1] if nxt else '-'}")
    pts_rows, price_rows, skipped = [], [], set()
    for g, meeting, locked, start in done + ([nxt] if nxt else []):
        feed = get(f"drivers/{g}_en.json", refresh=not locked)
        if not feed:
            print(f"  gameday {g} {meeting}: no feed")
            continue
        rnd = raced_rounds[g - 1]                      # our calendar round of this raced weekend
        prev_rnd = raced_rounds[g - 2] if g > 1 else 0
        for p in feed["Data"]["Value"]:
            name = name_of(p)
            typ = "driver" if p["PositionName"].upper().startswith("DRIVER") else "constructor"
            # the game makes a separate asset per (driver, team) - e.g. "Lawson (Red Bull Racing)" for his stand-in
            # rounds and Tsunoda at Racing Bulls; the model's driver = the asset of his 2026 team
            if typ == "driver" and GRID_2026.get(name) != canon_team(p["TeamName"]):
                skipped.add(f"{p['FUllName']} ({p['TeamName']})")
                continue
            # price paid for gameday g = price after the previous raced round
            price_rows.append({"round": prev_rnd, "gameday": g, "date": start[:10], "name": name, "type": typ,
                               "price": float(p["Value"])})
            if locked:
                pts_rows.append({"round": rnd, "gameday": g, "name": name, "type": typ,
                                 "points": float(p["GamedayPoints"] or 0), "dotd": 0.0, "fastest_lap": 0.0, "overtaking": 0.0})
        print(f"  gameday {g:2d} {meeting:28s} -> round {rnd:2d}  {len(feed['Data']['Value'])} assets")
    if skipped:
        print("  not model assets (stand-in / second-team entries), skipped:", sorted(skipped))
    pts = pd.DataFrame(pts_rows)
    # AdditionalStats (dotd_pts, ...) are season-to-date totals whose snapshots are NOT consistent between gameday
    # files (differencing them gives e.g. 90 Driver-of-the-Day points one weekend and none for eight), so they are
    # not used; GamedayPoints already includes Driver of the Day and pit-stop points.
    if len(pts):
        pts = pts.drop(columns=["dotd", "fastest_lap", "overtaking"])
        pts = pts.sort_values(["gameday", "type", "name"])
    prices = pd.DataFrame(price_rows)
    pts.to_csv("fantasy_f1_official_points.csv", index=False)
    prices.to_csv("fantasy_f1_prices_official.csv", index=False)
    print(f"wrote fantasy_f1_official_points.csv ({len(pts)} rows), fantasy_f1_prices_official.csv ({len(prices)} rows)")


if __name__ == "__main__":
    main()
