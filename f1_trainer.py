#!/usr/bin/env python3
"""
F1 Fantasy ML Trainer (f1_trainer.py)
=====================================
Collects race data from OpenF1 API for 2024-2025 seasons,
calculates comprehensive Fantasy F1 points, engineers features
(including lag features from prior year and preseason testing),
and trains a gradient boosted (XGBoost) model to predict
fantasy points for the 2026 season.

Usage:
    source venv/bin/activate
    python f1_trainer.py
"""

import json
import os
import time
import pickle
import concurrent.futures

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from openf1_utils import openf1_client

# ──────────────────────────────────────────────────────────────
# Fantasy F1 2026 Scoring Rules
# ──────────────────────────────────────────────────────────────

# Qualifying points (driver)
QUALI_PTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}

# Race finish points (driver)
RACE_PTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Sprint finish points (driver)
SPRINT_PTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

# Penalties and bonuses (driver)
DNF_PENALTY_RACE = -20
DNF_PENALTY_SPRINT = -10
DSQ_PENALTY = -20
QUALI_NC_PENALTY = -5
OVERTAKE_BONUS = 1          # per position overtaken in race
POSITION_GAINED = 1         # per grid position gained (quali -> race)
POSITION_LOST = -1          # per grid position lost (quali -> race)
FASTEST_LAP_RACE = 10
FASTEST_LAP_SPRINT = 5
DRIVER_OF_THE_DAY = 10      # not reliably in OpenF1, skip for training

# Constructor scoring
# Constructors score the combined total of both drivers' points
# (qualifying, race, sprint — but NOT Driver of the Day).
# Additional constructor-specific bonuses:
CONSTRUCTOR_QUALI_Q2_Q3 = {
    "neither_q2": -1,
    "one_q2": 1,
    "both_q2": 3,
    "one_q3": 5,
    "both_q3": 10,
}
CONSTRUCTOR_DSQ_PENALTY_QUALI = -5    # per disqualified driver in qualifying
CONSTRUCTOR_DSQ_PENALTY_SPRINT = -10  # per disqualified driver in sprint
CONSTRUCTOR_DSQ_PENALTY_RACE = -20    # per disqualified driver in race

# Pit stop scoring (constructor, based on stationary stop time)
PIT_STOP_TIERS = [
    (0.0, 2.0, 20),    # under 2.0s
    (2.0, 2.2, 10),    # 2.00 - 2.19s
    (2.2, 2.5, 5),     # 2.20 - 2.49s
    (2.5, 3.0, 2),     # 2.50 - 2.99s
    (3.0, float('inf'), 0),  # over 3.0s
]
PIT_STOP_FASTEST_BONUS = 5     # fastest pit stop of the race
PIT_STOP_WORLD_RECORD_BONUS = 15  # new world record (current record 1.8s)
PIT_STOP_WORLD_RECORD_TIME = 1.8  # current world record in seconds

# ──────────────────────────────────────────────────────────────
# 2026 Grid - Driver/Team Mapping
# ──────────────────────────────────────────────────────────────

GRID_2026 = {
    "Verstappen": "Red Bull Racing",
    "Hadjar": "Red Bull Racing",
    "Russell": "Mercedes",
    "Antonelli": "Mercedes",
    "Norris": "McLaren",
    "Piastri": "McLaren",
    "Leclerc": "Ferrari",
    "Hamilton": "Ferrari",
    "Gasly": "Alpine",
    "Colapinto": "Alpine",
    "Sainz": "Williams",
    "Albon": "Williams",
    "Alonso": "Aston Martin",
    "Stroll": "Aston Martin",
    "Bearman": "Haas F1 Team",
    "Ocon": "Haas F1 Team",
    "Hulkenberg": "Audi",
    "Bortoleto": "Audi",
    "Lawson": "Racing Bulls",
    "Lindblad": "Racing Bulls",
    "Perez": "Cadillac",
    "Bottas": "Cadillac",
}


# ──────────────────────────────────────────────────────────────
# Data Collection from OpenF1
# ──────────────────────────────────────────────────────────────

def openf1_get(endpoint, params=None, retries=3):
    return openf1_client.get(endpoint, params, retries=retries)


def get_session_data(session, driver_map):
    """Fetch position, lap, and stint data for a session."""
    sk = session["session_key"]
    stype = session["session_type"]
    if stype not in ("Qualifying", "Race", "Sprint", "Sprint Qualifying"):
        return None

    print(f"      Fetching: {stype} (key={sk})")
    time.sleep(0.3)

    data = {
        "session_key": sk,
        "session_type": stype,
        "positions": openf1_get("position", {"session_key": sk}),
    }

    if stype in ("Race", "Sprint"):
        data["laps"] = openf1_get("laps", {"session_key": sk})
        data["stints"] = openf1_get("stints", {"session_key": sk})
        data["pit"] = openf1_get("pit", {"session_key": sk})
    else:
        data["laps"] = []
        data["stints"] = []
        data["pit"] = []

    return data


def collect_data(years=None, output_file="raw_historical_data.json"):
    """Collect raw race data from OpenF1 for given years."""
    if years is None:
        years = [2023,2024, 2025]

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_data = json.load(f)
    else:
        all_data = []

    processed_keys = {m["meeting_key"] for m in all_data}

    openf1_client.check_refresh()
    is_paid = openf1_client.token is not None
    workers = 8 if is_paid else 3
    print(f"  Collection mode: {'PAID' if is_paid else 'FREE'} ({workers} workers)")

    for year in years:
        print(f"\n  ── {year} Season ──")
        meetings = openf1_get("meetings", {"year": year})
        if not meetings:
            continue

        for i, meeting in enumerate(meetings):
            mk = meeting["meeting_key"]
            location = meeting.get("location", "Unknown")
            if mk in processed_keys:
                continue

            print(f"    [{i+1}/{len(meetings)}] {location} (key={mk})")

            sessions = openf1_get("sessions", {"meeting_key": mk})
            drivers = []
            if sessions:
                drivers = openf1_get("drivers", {"session_key": sessions[0]["session_key"]})

            driver_map = {d["driver_number"]: d.get("last_name", "") for d in drivers}

            meeting_data = {
                "year": year,
                "location": location,
                "meeting_key": mk,
                "meeting_name": meeting.get("meeting_name", ""),
                "drivers": drivers,
                "sessions": [],
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(get_session_data, s, driver_map): s for s in sessions}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        meeting_data["sessions"].append(result)

            all_data.append(meeting_data)
            processed_keys.add(mk)

            with open(output_file, "w") as f:
                json.dump(all_data, f)

            time.sleep(0.1)

    return all_data


# ──────────────────────────────────────────────────────────────
# Processing: Raw Data -> Fantasy Points
# ──────────────────────────────────────────────────────────────

def _latest_positions(positions_list):
    """From a list of position records, return the latest position per driver."""
    latest = {}
    for p in positions_list:
        dn = p.get("driver_number")
        pos = p.get("position")
        if dn is not None and pos is not None:
            latest[dn] = pos
    return latest


def _compute_overtakes(laps_list, driver_map):
    """Count net overtakes per driver from lap-by-lap position data."""
    driver_laps = {}
    for lap in laps_list:
        dn = lap.get("driver_number")
        lap_num = lap.get("lap_number")
        pos = lap.get("position")
        if dn and lap_num and pos:
            driver_laps.setdefault(dn, []).append((lap_num, pos))

    overtakes = {}
    for dn, laps in driver_laps.items():
        laps.sort()
        ov = 0
        for i in range(1, len(laps)):
            diff = laps[i - 1][1] - laps[i][1]
            if diff > 0:
                ov += diff
        name = driver_map.get(dn, str(dn))
        overtakes[name] = ov

    return overtakes


def _find_fastest_lap(laps_list, driver_map):
    """Find the driver with the fastest lap."""
    fastest_time = None
    fastest_dn = None
    for lap in laps_list:
        dur = lap.get("lap_duration")
        dn = lap.get("driver_number")
        if dur and dn:
            if fastest_time is None or dur < fastest_time:
                fastest_time = dur
                fastest_dn = dn
    if fastest_dn:
        return driver_map.get(fastest_dn)
    return None


def _detect_dnf(stints_list, driver_map):
    """Detect DNFs based on stint data - drivers who completed < 90% of max laps."""
    if not stints_list:
        return set()
    driver_max_lap = {}
    overall_max = 0
    for s in stints_list:
        dn = s.get("driver_number")
        end = s.get("lap_end", 0) or 0
        if dn:
            driver_max_lap[dn] = max(driver_max_lap.get(dn, 0), end)
            overall_max = max(overall_max, end)

    dnf_drivers = set()
    for dn, laps in driver_max_lap.items():
        if laps < overall_max * 0.9:
            name = driver_map.get(dn, str(dn))
            dnf_drivers.add(name)
    return dnf_drivers


def _fastest_pit_stop(pit_list, driver_map, team_map):
    """Find fastest and slowest pit stop teams."""
    if not pit_list:
        return None, None
    team_best = {}
    for p in pit_list:
        dur = p.get("pit_duration")
        dn = p.get("driver_number")
        if dur and dn:
            team = team_map.get(dn, "Unknown")
            if team not in team_best or dur < team_best[team]:
                team_best[team] = dur

    if not team_best:
        return None, None
    fastest_team = min(team_best, key=team_best.get)
    slowest_team = max(team_best, key=team_best.get)
    return fastest_team, slowest_team


def process_raw_data(data, years_filter=None):
    """Process raw OpenF1 data into a DataFrame with fantasy points per driver per race."""
    driver_rows = []
    constructor_rows = []

    for meeting in data:
        year = meeting["year"]
        if years_filter and year not in years_filter:
            continue

        location = meeting["location"]
        drivers_list = meeting.get("drivers", [])
        driver_map = {d["driver_number"]: d.get("last_name", "") for d in drivers_list}
        team_map = {d["driver_number"]: d.get("team_name", "Unknown") for d in drivers_list}

        # Build name->team mapping
        name_team = {}
        for d in drivers_list:
            name_team[d.get("last_name", "")] = d.get("team_name", "Unknown")

        # Collect session results
        quali_pos = {}
        race_pos = {}
        sprint_quali_pos = {}
        sprint_pos = {}
        race_overtakes = {}
        sprint_overtakes = {}
        race_fastest_lap = None
        sprint_fastest_lap = None
        race_dnf = set()
        sprint_dnf = set()
        is_sprint = False
        fastest_pit_team = None
        slowest_pit_team = None
        race_pit_data = []

        # ── Re-label sprint sessions ──
        # OpenF1 often labels sprint sessions as plain "Qualifying"/"Race".
        # Sprint weekends have 4 sessions; the lower session_key pair is the sprint.
        raw_sessions = sorted(meeting.get("sessions", []), key=lambda s: s["session_key"])
        quali_sessions = [s for s in raw_sessions if s["session_type"] == "Qualifying"]
        race_sessions = [s for s in raw_sessions if s["session_type"] == "Race"]
        if len(quali_sessions) == 2 and len(race_sessions) == 2:
            # First (lower key) of each pair is the sprint variant
            quali_sessions[0]["session_type"] = "Sprint Qualifying"
            race_sessions[0]["session_type"] = "Sprint"

        for session in raw_sessions:
            stype = session["session_type"]
            positions = session.get("positions", [])
            latest = _latest_positions(positions)

            if stype == "Qualifying":
                for dn, pos in latest.items():
                    name = driver_map.get(dn)
                    if name:
                        quali_pos[name] = pos

            elif stype == "Race":
                for dn, pos in latest.items():
                    name = driver_map.get(dn)
                    if name:
                        race_pos[name] = pos
                race_overtakes = _compute_overtakes(session.get("laps", []), driver_map)
                race_fastest_lap = _find_fastest_lap(session.get("laps", []), driver_map)
                race_dnf = _detect_dnf(session.get("stints", []), driver_map)
                race_pit_data = session.get("pit", [])
                fastest_pit_team, slowest_pit_team = _fastest_pit_stop(
                    race_pit_data, driver_map, team_map
                )

            elif stype == "Sprint Qualifying":
                is_sprint = True
                for dn, pos in latest.items():
                    name = driver_map.get(dn)
                    if name:
                        sprint_quali_pos[name] = pos

            elif stype == "Sprint":
                is_sprint = True
                for dn, pos in latest.items():
                    name = driver_map.get(dn)
                    if name:
                        sprint_pos[name] = pos
                sprint_overtakes = _compute_overtakes(session.get("laps", []), driver_map)
                sprint_fastest_lap = _find_fastest_lap(session.get("laps", []), driver_map)
                sprint_dnf = _detect_dnf(session.get("stints", []), driver_map)

        # Skip meetings with no competitive data (e.g., testing)
        if not quali_pos and not race_pos:
            continue

        # ── Calculate driver fantasy points ──
        team_driver_race_pos = {}  # team -> list of race positions

        for name in driver_map.values():
            if not name:
                continue
            team = name_team.get(name, "Unknown")
            pts = 0

            # Qualifying
            qp = quali_pos.get(name)
            if isinstance(qp, int):
                pts += QUALI_PTS.get(qp, 0)

            # Race
            rp = race_pos.get(name)
            if name in race_dnf:
                pts += DNF_PENALTY_RACE
            elif isinstance(rp, int):
                pts += RACE_PTS.get(rp, 0)
                # Position change bonus/penalty
                if isinstance(qp, int):
                    gain = qp - rp  # positive = gained positions
                    if gain > 0:
                        pts += gain * POSITION_GAINED
                    elif gain < 0:
                        pts += abs(gain) * POSITION_LOST

                # Track race position for constructor bonus
                team_driver_race_pos.setdefault(team, []).append(rp)

            # Overtakes
            pts += race_overtakes.get(name, 0) * OVERTAKE_BONUS

            # Fastest lap
            if race_fastest_lap == name:
                pts += FASTEST_LAP_RACE

            # Sprint weekend
            if is_sprint:
                sp = sprint_pos.get(name)
                if name in sprint_dnf:
                    pts += DNF_PENALTY_SPRINT
                elif isinstance(sp, int):
                    pts += SPRINT_PTS.get(sp, 0)
                    # Position change (sprint grid vs sprint finish)
                    sqp = sprint_quali_pos.get(name)
                    if isinstance(sqp, int):
                        sgain = sqp - sp
                        if sgain > 0:
                            pts += sgain * POSITION_GAINED
                        elif sgain < 0:
                            pts += abs(sgain) * POSITION_LOST

                if sprint_fastest_lap == name:
                    pts += FASTEST_LAP_SPRINT

                pts += sprint_overtakes.get(name, 0) * OVERTAKE_BONUS

            driver_rows.append({
                "year": year,
                "location": location,
                "driver": name,
                "team": team,
                "fantasy_points": pts,
                "quali_pos": qp if isinstance(qp, int) else None,
                "race_pos": rp if isinstance(rp, int) and name not in race_dnf else None,
                "dnf": 1 if name in race_dnf else 0,
                "overtakes": race_overtakes.get(name, 0),
                "fastest_lap": 1 if race_fastest_lap == name else 0,
                "is_sprint_weekend": 1 if is_sprint else 0,
                "sprint_pos": sprint_pos.get(name) if isinstance(sprint_pos.get(name), int) else None,
            })

        # ── Calculate constructor fantasy points ──
        # Constructors score the combined total of both drivers' points
        # (excluding Driver of the Day), plus constructor-specific bonuses.
        teams_in_meeting = set(name_team.values())
        for team in teams_in_meeting:
            if team == "Unknown":
                continue
            c_pts = 0
            team_drivers = [n for n, t in name_team.items() if t == team]

            # ── Qualifying: combined driver qualifying points ──
            for d in team_drivers:
                qp = quali_pos.get(d)
                if isinstance(qp, int):
                    c_pts += QUALI_PTS.get(qp, 0)
                else:
                    c_pts += QUALI_NC_PENALTY  # NC/DSQ/no time

            # Q2/Q3 progression bonus (infer from final quali position)
            # Top 15 made Q2, top 10 made Q3
            team_quali_positions = [quali_pos.get(d) for d in team_drivers
                                    if isinstance(quali_pos.get(d), int)]
            q3_count = sum(1 for p in team_quali_positions if p <= 10)
            q2_count = sum(1 for p in team_quali_positions if p <= 15)
            if q3_count >= 2:
                c_pts += CONSTRUCTOR_QUALI_Q2_Q3["both_q3"]
            elif q3_count == 1:
                c_pts += CONSTRUCTOR_QUALI_Q2_Q3["one_q3"]
            elif q2_count >= 2:
                c_pts += CONSTRUCTOR_QUALI_Q2_Q3["both_q2"]
            elif q2_count == 1:
                c_pts += CONSTRUCTOR_QUALI_Q2_Q3["one_q2"]
            else:
                c_pts += CONSTRUCTOR_QUALI_Q2_Q3["neither_q2"]

            # ── Race: combined driver race points ──
            for d in team_drivers:
                if d in race_dnf:
                    c_pts += DNF_PENALTY_RACE
                else:
                    rp = race_pos.get(d)
                    if isinstance(rp, int):
                        c_pts += RACE_PTS.get(rp, 0)
                        # Position change bonus/penalty
                        qp = quali_pos.get(d)
                        if isinstance(qp, int):
                            gain = qp - rp
                            if gain > 0:
                                c_pts += gain * POSITION_GAINED
                            elif gain < 0:
                                c_pts += abs(gain) * POSITION_LOST

                # Overtakes (race)
                c_pts += race_overtakes.get(d, 0) * OVERTAKE_BONUS

            # Race fastest lap (constructors get this too)
            if race_fastest_lap and race_fastest_lap in team_drivers:
                c_pts += FASTEST_LAP_RACE

            # ── Pit stop scoring (tiered by stop duration) ──
            team_pit_stops = [p for p in race_pit_data
                              if team_map.get(p.get("driver_number")) == team]
            for p in team_pit_stops:
                dur = p.get("stop_duration") or p.get("pit_duration")
                if dur and isinstance(dur, (int, float)):
                    for low, high, tier_pts in PIT_STOP_TIERS:
                        if low <= dur < high:
                            c_pts += tier_pts
                            break
                    if dur < PIT_STOP_WORLD_RECORD_TIME:
                        c_pts += PIT_STOP_WORLD_RECORD_BONUS

            # Fastest pit stop bonus
            if fastest_pit_team == team:
                c_pts += PIT_STOP_FASTEST_BONUS

            # ── Sprint: combined driver sprint points ──
            if is_sprint:
                for d in team_drivers:
                    if d in sprint_dnf:
                        c_pts += DNF_PENALTY_SPRINT
                    else:
                        sp = sprint_pos.get(d)
                        if isinstance(sp, int):
                            c_pts += SPRINT_PTS.get(sp, 0)
                            # Sprint position change
                            sqp = sprint_quali_pos.get(d)
                            if isinstance(sqp, int):
                                sgain = sqp - sp
                                if sgain > 0:
                                    c_pts += sgain * POSITION_GAINED
                                elif sgain < 0:
                                    c_pts += abs(sgain) * POSITION_LOST

                    # Sprint overtakes
                    c_pts += sprint_overtakes.get(d, 0) * OVERTAKE_BONUS

                # Sprint fastest lap (constructors get this too)
                if sprint_fastest_lap and sprint_fastest_lap in team_drivers:
                    c_pts += FASTEST_LAP_SPRINT

            constructor_rows.append({
                "year": year,
                "location": location,
                "team": team,
                "constructor_fantasy_points": c_pts,
                "is_sprint_weekend": 1 if is_sprint else 0,
            })

    driver_df = pd.DataFrame(driver_rows)
    constructor_df = pd.DataFrame(constructor_rows)
    return driver_df, constructor_df


# ──────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────

def add_round_numbers(df):
    """Add sequential round numbers within each year based on appearance order."""
    if df.empty:
        return df
    df = df.copy()
    locations_by_year = (
        df.drop_duplicates(subset=["year", "location"])
        .sort_values(["year", "location"])
        .groupby("year")["location"]
        .apply(list)
        .to_dict()
    )
    round_map = {}
    for year, locs in locations_by_year.items():
        for i, loc in enumerate(locs, 1):
            round_map[(year, loc)] = i

    df["round"] = df.apply(lambda r: round_map.get((r["year"], r["location"]), 0), axis=1)
    return df


def engineer_driver_features(df):
    """Create lag and rolling features for driver fantasy points prediction."""
    df = df.copy()
    df = add_round_numbers(df)
    df = df.sort_values(["driver", "year", "round"])

    # Lag features (shifted within driver, across all years)
    df["prev_points"] = df.groupby("driver")["fantasy_points"].shift(1)
    df["avg_points_3"] = (
        df.groupby("driver")["fantasy_points"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    df["avg_points_5"] = (
        df.groupby("driver")["fantasy_points"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    )
    df["max_points_3"] = (
        df.groupby("driver")["fantasy_points"]
        .transform(lambda x: x.rolling(3, min_periods=1).max().shift(1))
    )
    df["std_points_3"] = (
        df.groupby("driver")["fantasy_points"]
        .transform(lambda x: x.rolling(3, min_periods=1).std().shift(1))
    )

    # Season-level stats from prior year
    yearly_avg = df.groupby(["driver", "year"])["fantasy_points"].mean().reset_index()
    yearly_avg.columns = ["driver", "year", "yearly_avg_pts"]
    yearly_avg["year"] = yearly_avg["year"] + 1  # shift to become "prior year" feature
    df = df.merge(yearly_avg, on=["driver", "year"], how="left")

    yearly_best = df.groupby(["driver", "year"])["fantasy_points"].max().reset_index()
    yearly_best.columns = ["driver", "year", "yearly_best_pts"]
    yearly_best["year"] = yearly_best["year"] + 1
    df = df.merge(yearly_best, on=["driver", "year"], how="left")

    # Quali and race position lags
    df["prev_quali"] = df.groupby("driver")["quali_pos"].shift(1)
    df["prev_race_pos"] = df.groupby("driver")["race_pos"].shift(1)
    df["prev_dnf"] = df.groupby("driver")["dnf"].shift(1)

    # Track-specific historical performance
    df["track_avg_pts"] = (
        df.groupby(["driver", "location"])["fantasy_points"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # Fill NaN with sensible defaults
    fill_cols = [
        "prev_points", "avg_points_3", "avg_points_5", "max_points_3",
        "std_points_3", "yearly_avg_pts", "yearly_best_pts",
        "prev_quali", "prev_race_pos", "prev_dnf", "track_avg_pts"
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    return df


def engineer_constructor_features(df):
    """Create lag and rolling features for constructor fantasy points."""
    df = df.copy()
    df = add_round_numbers(df)
    df = df.sort_values(["team", "year", "round"])

    df["c_prev_points"] = df.groupby("team")["constructor_fantasy_points"].shift(1)
    df["c_avg_points_3"] = (
        df.groupby("team")["constructor_fantasy_points"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    df["c_avg_points_5"] = (
        df.groupby("team")["constructor_fantasy_points"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    )
    df["c_max_points_3"] = (
        df.groupby("team")["constructor_fantasy_points"]
        .transform(lambda x: x.rolling(3, min_periods=1).max().shift(1))
    )
    df["c_std_points_3"] = (
        df.groupby("team")["constructor_fantasy_points"]
        .transform(lambda x: x.rolling(3, min_periods=1).std().shift(1))
    )

    yearly_avg = df.groupby(["team", "year"])["constructor_fantasy_points"].mean().reset_index()
    yearly_avg.columns = ["team", "year", "c_yearly_avg_pts"]
    yearly_avg["year"] = yearly_avg["year"] + 1
    df = df.merge(yearly_avg, on=["team", "year"], how="left")

    yearly_best = df.groupby(["team", "year"])["constructor_fantasy_points"].max().reset_index()
    yearly_best.columns = ["team", "year", "c_yearly_best_pts"]
    yearly_best["year"] = yearly_best["year"] + 1
    df = df.merge(yearly_best, on=["team", "year"], how="left")

    df["c_track_avg"] = (
        df.groupby(["team", "location"])["constructor_fantasy_points"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    fill_cols = [
        "c_prev_points", "c_avg_points_3", "c_avg_points_5",
        "c_max_points_3", "c_std_points_3",
        "c_yearly_avg_pts", "c_yearly_best_pts", "c_track_avg",
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    return df


# ──────────────────────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────────────────────

DRIVER_FEATURES = [
    "year", "round", "driver_enc", "team_enc", "loc_enc",
    "prev_points", "avg_points_3", "avg_points_5", "max_points_3", "std_points_3",
    "yearly_avg_pts", "yearly_best_pts",
    "prev_quali", "prev_race_pos", "prev_dnf",
    "track_avg_pts", "is_sprint_weekend",
]

CONSTRUCTOR_FEATURES = [
    "year", "round", "team_enc", "loc_enc",
    "c_prev_points", "c_avg_points_3", "c_avg_points_5",
    "c_max_points_3", "c_std_points_3",
    "c_yearly_avg_pts", "c_yearly_best_pts", "c_track_avg",
    "is_sprint_weekend",
]


def train_driver_model(df):
    """Train XGBoost model for driver fantasy points."""
    if df.empty:
        print("  ⚠ Empty driver dataset. Aborting.")
        return None

    df = df.copy()

    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_loc = LabelEncoder()

    df["driver_enc"] = le_driver.fit_transform(df["driver"])
    df["team_enc"] = le_team.fit_transform(df["team"].astype(str))
    df["loc_enc"] = le_loc.fit_transform(df["location"])

    X = df[DRIVER_FEATURES]
    y = df["fantasy_points"]

    # Time-series aware split: train on earlier data, test on later
    tscv = TimeSeriesSplit(n_splits=3)
    best_mae = float("inf")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    # Evaluate with time-series CV
    maes = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)

    print(f"  Driver Model CV MAE: {np.mean(maes):.2f} (+/- {np.std(maes):.2f})")

    # Final fit on all data
    model.fit(X, y, verbose=False)

    # Feature importance
    importance = dict(zip(DRIVER_FEATURES, model.feature_importances_))
    print("  Top features:", sorted(importance.items(), key=lambda x: -x[1])[:5])

    artifacts = {
        "model": model,
        "le_driver": le_driver,
        "le_team": le_team,
        "le_loc": le_loc,
        "features": DRIVER_FEATURES,
        "driver_df": df[["driver", "team", "year", "round", "location", "fantasy_points",
                          "quali_pos", "race_pos", "dnf", "overtakes",
                          "prev_points", "avg_points_3", "avg_points_5",
                          "max_points_3", "std_points_3",
                          "yearly_avg_pts", "yearly_best_pts",
                          "prev_quali", "prev_race_pos", "prev_dnf",
                          "track_avg_pts", "is_sprint_weekend"]].copy(),
    }

    with open("f1_driver_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("  ✅ Driver model saved to f1_driver_model.pkl")
    return artifacts


def train_constructor_model(df):
    """Train XGBoost model for constructor fantasy points."""
    if df.empty:
        print("  ⚠ Empty constructor dataset. Aborting.")
        return None

    df = df.copy()

    le_team = LabelEncoder()
    le_loc = LabelEncoder()

    df["team_enc"] = le_team.fit_transform(df["team"].astype(str))
    df["loc_enc"] = le_loc.fit_transform(df["location"])

    X = df[CONSTRUCTOR_FEATURES]
    y = df["constructor_fantasy_points"]

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    maes = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))

    print(f"  Constructor Model CV MAE: {np.mean(maes):.2f} (+/- {np.std(maes):.2f})")

    model.fit(X, y, verbose=False)

    artifacts = {
        "model": model,
        "le_team": le_team,
        "le_loc": le_loc,
        "features": CONSTRUCTOR_FEATURES,
        "constructor_df": df[["team", "year", "round", "location",
                               "constructor_fantasy_points",
                               "c_prev_points", "c_avg_points_3", "c_avg_points_5",
                               "c_max_points_3", "c_std_points_3",
                               "c_yearly_avg_pts", "c_yearly_best_pts", "c_track_avg",
                               "is_sprint_weekend"]].copy(),
    }

    with open("f1_constructor_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("  ✅ Constructor model saved to f1_constructor_model.pkl")
    return artifacts


# ──────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  F1 Fantasy ML Training Pipeline")
    print("=" * 60)

    # Step 1: Collect data
    print("\n📡 Step 1: Collecting data from OpenF1 (2024-2025)...")
    data = collect_data(years=[2024, 2025])

    # Step 2: Process into fantasy points
    print("\n📊 Step 2: Calculating fantasy points...")
    driver_df, constructor_df = process_raw_data(data, years_filter=[2024, 2025])

    if driver_df.empty:
        print("  ⚠ No driver data found. Check raw_historical_data.json")
        return

    driver_df.to_csv("f1_driver_training_data.csv", index=False)
    constructor_df.to_csv("f1_constructor_training_data.csv", index=False)
    print(f"  Driver records: {len(driver_df)}")
    print(f"  Constructor records: {len(constructor_df)}")

    # Step 3: Feature engineering
    print("\n🔧 Step 3: Engineering features...")
    driver_df = engineer_driver_features(driver_df)
    constructor_df = engineer_constructor_features(constructor_df)

    # Step 4: Train models
    print("\n🧠 Step 4: Training gradient boosted models...")
    print("\n  ── Driver Model ──")
    train_driver_model(driver_df)

    print("\n  ── Constructor Model ──")
    train_constructor_model(constructor_df)

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
