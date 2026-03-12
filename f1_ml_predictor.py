#!/usr/bin/env python3
"""
F1 Fantasy ML Predictor (f1_ml_predictor.py)
=============================================
Loads trained XGBoost models and predicts fantasy points
for drivers and constructors at a given 2026 race.

From the *second* race onwards, the predictor automatically fetches
actual 2026 qualifying/race results from OpenF1 and uses them to
update lag features (rolling averages, previous result, etc.) so that
each prediction reflects what really happened in earlier rounds.
"""

import os
import json
import pickle
import time

import pandas as pd
import numpy as np

from f1_trainer import (
    GRID_2026, process_raw_data, add_round_numbers,
    openf1_get,
)
from openf1_utils import openf1_client

# ──────────────────────────────────────────────────────────────
# 2026 Race Calendar (location names matching OpenF1 conventions)
# ──────────────────────────────────────────────────────────────

RACE_CALENDAR_2026 = [
    {"round": 1, "location": "Melbourne", "name": "Australian GP", "sprint": False},
    {"round": 2, "location": "Shanghai", "name": "Chinese GP", "sprint": True},
    {"round": 3, "location": "Suzuka", "name": "Japanese GP", "sprint": False},
    {"round": 4, "location": "Sakhir", "name": "Bahrain GP", "sprint": False},
    {"round": 5, "location": "Jeddah", "name": "Saudi Arabian GP", "sprint": False},
    {"round": 6, "location": "Miami", "name": "Miami GP", "sprint": True},
    {"round": 7, "location": "Montréal", "name": "Canadian GP", "sprint": True},
    {"round": 8, "location": "Monaco", "name": "Monaco GP", "sprint": False},
    {"round": 9, "location": "Barcelona", "name": "Barcelona-Catalunya GP", "sprint": False},
    {"round": 10, "location": "Spielberg", "name": "Austrian GP", "sprint": False},
    {"round": 11, "location": "Silverstone", "name": "British GP", "sprint": True},
    {"round": 12, "location": "Spa-Francorchamps", "name": "Belgian GP", "sprint": False},
    {"round": 13, "location": "Budapest", "name": "Hungarian GP", "sprint": False},
    {"round": 14, "location": "Zandvoort", "name": "Dutch GP", "sprint": True},
    {"round": 15, "location": "Monza", "name": "Italian GP", "sprint": False},
    {"round": 16, "location": "Madrid", "name": "Spanish GP", "sprint": False},
    {"round": 17, "location": "Baku", "name": "Azerbaijan GP", "sprint": False},
    {"round": 18, "location": "Marina Bay", "name": "Singapore GP", "sprint": True},
    {"round": 19, "location": "Austin", "name": "US GP", "sprint": False},
    {"round": 20, "location": "Mexico City", "name": "Mexico City GP", "sprint": False},
    {"round": 21, "location": "São Paulo", "name": "São Paulo GP", "sprint": False},
    {"round": 22, "location": "Las Vegas", "name": "Las Vegas GP", "sprint": False},
    {"round": 23, "location": "Lusail", "name": "Qatar GP", "sprint": False},
    {"round": 24, "location": "Yas Island", "name": "Abu Dhabi GP", "sprint": False},
]


def get_race_info(round_num):
    """Get race info from the calendar by round number."""
    for race in RACE_CALENDAR_2026:
        if race["round"] == round_num:
            return race
    return None


# ──────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────

def load_driver_model(path="f1_driver_model.pkl"):
    if not os.path.exists(path):
        print(f"  ⚠ Driver model not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_constructor_model(path="f1_constructor_model.pkl"):
    if not os.path.exists(path):
        print(f"  ⚠ Constructor model not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────
# Fetch Actual 2026 Results from OpenF1
# ──────────────────────────────────────────────────────────────

CACHE_FILE_2026 = "raw_2026_results.json"


# Aliases for OpenF1 location names that differ from our calendar
_LOCATION_ALIASES = {
    "monte carlo": "Monaco",
    "yas marina": "Yas Island",
    "miami gardens": "Miami",
    "imola": "Barcelona",        # Imola no longer on 2026 calendar, closest match
    "le castellet": "Barcelona",
    "montreal": "Montréal",
}


def _round_for_location(location):
    """Map a location string to its 2026 round number (fuzzy + alias match)."""
    loc_lower = location.lower().strip()

    # Check aliases first
    for alias, canonical in _LOCATION_ALIASES.items():
        if alias in loc_lower:
            for race in RACE_CALENDAR_2026:
                if race["location"] == canonical:
                    return race["round"]

    # Substring match against calendar
    for race in RACE_CALENDAR_2026:
        cal = race["location"].lower()
        if cal in loc_lower or loc_lower in cal:
            return race["round"]
    return None


def fetch_2026_results():
    """
    Fetch all completed 2026 meeting data from OpenF1.
    Returns a list of meeting dicts in the same format as raw_historical_data.json.
    Results are cached to raw_2026_results.json.
    """
    # Load cache
    cached = []
    if os.path.exists(CACHE_FILE_2026):
        with open(CACHE_FILE_2026, "r") as f:
            cached = json.load(f)

    cached_keys = {m["meeting_key"] for m in cached}

    openf1_client.check_refresh()
    meetings = openf1_get("meetings", {"year": 2026})
    if not meetings:
        return cached

    from f1_trainer import get_session_data
    import concurrent.futures

    is_paid = openf1_client.token is not None
    workers = 8 if is_paid else 3

    new_meetings = []
    for meeting in meetings:
        mk = meeting["meeting_key"]
        if mk in cached_keys:
            continue

        location = meeting.get("location", "Unknown")
        sessions = openf1_get("sessions", {"meeting_key": mk})
        if not sessions:
            continue

        # Check if there's at least a completed Race session
        has_race = any(s.get("session_type") == "Race" for s in sessions)
        if not has_race:
            continue

        print(f"  📡 Fetching 2026 results: {location} (key={mk})")
        drivers = openf1_get("drivers", {"session_key": sessions[0]["session_key"]})
        driver_map = {d["driver_number"]: d.get("last_name", "") for d in drivers}

        meeting_data = {
            "year": 2026,
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

        new_meetings.append(meeting_data)
        time.sleep(0.1)

    if new_meetings:
        cached.extend(new_meetings)
        with open(CACHE_FILE_2026, "w") as f:
            json.dump(cached, f)
        print(f"  ✅ Cached {len(new_meetings)} new 2026 meeting(s)")

    return cached


def get_completed_2026_rounds(meetings_2026):
    """
    From fetched 2026 meeting data, determine which calendar rounds are complete.
    Returns a set of round numbers.
    """
    completed = set()
    for m in meetings_2026:
        rnd = _round_for_location(m["location"])
        if rnd is not None:
            completed.add(rnd)
    return completed


# ──────────────────────────────────────────────────────────────
# Build Combined Data with Actual 2026 Results
# ──────────────────────────────────────────────────────────────

def _load_historical_training_data():
    """Load the saved training CSVs from model training and add round numbers."""
    driver_df = None
    constructor_df = None
    if os.path.exists("f1_driver_training_data.csv"):
        driver_df = pd.read_csv("f1_driver_training_data.csv")
        driver_df = add_round_numbers(driver_df)
    if os.path.exists("f1_constructor_training_data.csv"):
        constructor_df = pd.read_csv("f1_constructor_training_data.csv")
        constructor_df = add_round_numbers(constructor_df)
    return driver_df, constructor_df


def build_combined_data(up_to_round):
    """
    Build combined driver/constructor DataFrames that include:
      - Historical 2024-2025 data (from training CSVs)
      - Actual 2026 results for rounds < up_to_round

    This allows lag features to reflect what really happened in the 2026 season.
    Returns (driver_df, constructor_df) with round numbers assigned.
    """
    hist_driver_df, hist_constructor_df = _load_historical_training_data()

    if up_to_round <= 1:
        # No prior 2026 data to incorporate
        return hist_driver_df, hist_constructor_df

    # Fetch actual 2026 results
    meetings_2026 = fetch_2026_results()
    if not meetings_2026:
        print("  ℹ No completed 2026 races found on OpenF1; using historical data only.")
        return hist_driver_df, hist_constructor_df

    # Filter to only rounds before the one we're predicting
    completed_rounds = get_completed_2026_rounds(meetings_2026)
    prior_rounds = {r for r in completed_rounds if r < up_to_round}

    if not prior_rounds:
        print(f"  ℹ No completed 2026 rounds before round {up_to_round}; using historical data only.")
        return hist_driver_df, hist_constructor_df

    # Filter meetings to only prior rounds
    prior_meetings = []
    for m in meetings_2026:
        rnd = _round_for_location(m["location"])
        if rnd in prior_rounds:
            prior_meetings.append(m)

    # Process 2026 meetings into fantasy point rows
    driver_2026, constructor_2026 = process_raw_data(prior_meetings, years_filter=[2026])

    if driver_2026.empty and constructor_2026.empty:
        return hist_driver_df, hist_constructor_df

    # Assign round numbers to 2026 data
    if not driver_2026.empty:
        driver_2026["round"] = driver_2026["location"].apply(
            lambda loc: _round_for_location(loc) or 0
        )
        driver_2026["year"] = 2026

    if not constructor_2026.empty:
        constructor_2026["round"] = constructor_2026["location"].apply(
            lambda loc: _round_for_location(loc) or 0
        )
        constructor_2026["year"] = 2026

    # Combine historical + 2026 actual
    combined_driver = hist_driver_df
    combined_constructor = hist_constructor_df

    if not driver_2026.empty and hist_driver_df is not None:
        # Ensure column alignment
        for col in hist_driver_df.columns:
            if col not in driver_2026.columns:
                driver_2026[col] = 0
        driver_2026 = driver_2026[hist_driver_df.columns]
        combined_driver = pd.concat([hist_driver_df, driver_2026], ignore_index=True)

    if not constructor_2026.empty and hist_constructor_df is not None:
        for col in hist_constructor_df.columns:
            if col not in constructor_2026.columns:
                constructor_2026[col] = 0
        constructor_2026 = constructor_2026[hist_constructor_df.columns]
        combined_constructor = pd.concat([hist_constructor_df, constructor_2026], ignore_index=True)

    completed_str = ", ".join(
        f"R{r}" for r in sorted(prior_rounds)
    )
    print(f"  ✅ Incorporated actual 2026 results: {completed_str}")

    return combined_driver, combined_constructor


# ──────────────────────────────────────────────────────────────
# Lag Feature Computation (Updated for live 2026 data)
# ──────────────────────────────────────────────────────────────

def get_driver_lag_features(driver_df, driver_name):
    """
    Compute lag features for a driver from the combined data.
    Returns a dict of feature values.
    """
    if driver_df is None or driver_df.empty:
        return _default_driver_lags()

    ddf = driver_df[driver_df["driver"] == driver_name].copy()
    if ddf.empty:
        return _default_driver_lags()

    ddf = ddf.sort_values(["year", "round"])
    last = ddf.iloc[-1]
    last3 = ddf.tail(3)
    last5 = ddf.tail(5)

    # Get the latest year's average (becomes yearly_avg_pts)
    latest_year = ddf["year"].max()
    year_data = ddf[ddf["year"] == latest_year]

    return {
        "prev_points": float(last["fantasy_points"]),
        "avg_points_3": float(last3["fantasy_points"].mean()),
        "avg_points_5": float(last5["fantasy_points"].mean()),
        "max_points_3": float(last3["fantasy_points"].max()),
        "std_points_3": float(last3["fantasy_points"].std()) if len(last3) > 1 else 0.0,
        "yearly_avg_pts": float(year_data["fantasy_points"].mean()),
        "yearly_best_pts": float(year_data["fantasy_points"].max()),
        "prev_quali": float(last["quali_pos"]) if pd.notna(last.get("quali_pos")) else 10.0,
        "prev_race_pos": float(last["race_pos"]) if pd.notna(last.get("race_pos")) else 10.0,
        "prev_dnf": float(last.get("dnf", 0)),
    }


def get_driver_track_avg(driver_df, driver_name, location):
    """Get a driver's historical average at a specific track."""
    if driver_df is None or driver_df.empty:
        return 0.0
    track_data = driver_df[(driver_df["driver"] == driver_name) & (driver_df["location"] == location)]
    if track_data.empty:
        return 0.0
    return float(track_data["fantasy_points"].mean())


def _default_driver_lags():
    return {
        "prev_points": 0.0, "avg_points_3": 0.0, "avg_points_5": 0.0,
        "max_points_3": 0.0, "std_points_3": 0.0,
        "yearly_avg_pts": 0.0, "yearly_best_pts": 0.0,
        "prev_quali": 15.0, "prev_race_pos": 15.0, "prev_dnf": 0.0,
    }


def get_constructor_lag_features(constructor_df, team_name):
    """Compute lag features for a constructor."""
    if constructor_df is None or constructor_df.empty:
        return _default_constructor_lags()

    tdf = constructor_df[constructor_df["team"] == team_name].copy()
    if tdf.empty:
        return _default_constructor_lags()

    tdf = tdf.sort_values(["year", "round"])
    last = tdf.iloc[-1]
    last3 = tdf.tail(3)
    last5 = tdf.tail(5)

    latest_year = tdf["year"].max()
    year_data = tdf[tdf["year"] == latest_year]

    return {
        "c_prev_points": float(last["constructor_fantasy_points"]),
        "c_avg_points_3": float(last3["constructor_fantasy_points"].mean()),
        "c_avg_points_5": float(last5["constructor_fantasy_points"].mean()),
        "c_max_points_3": float(last3["constructor_fantasy_points"].max()),
        "c_std_points_3": float(last3["constructor_fantasy_points"].std()) if len(last3) > 1 else 0.0,
        "c_yearly_avg_pts": float(year_data["constructor_fantasy_points"].mean()),
        "c_yearly_best_pts": float(year_data["constructor_fantasy_points"].max()),
    }


def get_constructor_track_avg(constructor_df, team_name, location):
    if constructor_df is None or constructor_df.empty:
        return 0.0
    track_data = constructor_df[(constructor_df["team"] == team_name) & (constructor_df["location"] == location)]
    if track_data.empty:
        return 0.0
    return float(track_data["constructor_fantasy_points"].mean())


def _default_constructor_lags():
    return {
        "c_prev_points": 0.0, "c_avg_points_3": 0.0,
        "c_avg_points_5": 0.0, "c_max_points_3": 0.0,
        "c_std_points_3": 0.0, "c_yearly_avg_pts": 0.0,
        "c_yearly_best_pts": 0.0,
    }


# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────

def predict_driver_scores(round_num, year=2026):
    """Predict fantasy points for all 2026 drivers at a given race."""
    race_info = get_race_info(round_num)
    if not race_info:
        print(f"  ⚠ Round {round_num} not found in calendar")
        return {}

    artifacts = load_driver_model()
    if not artifacts:
        return {}

    model = artifacts["model"]
    le_driver = artifacts["le_driver"]
    le_team = artifacts["le_team"]
    le_loc = artifacts["le_loc"]
    features = artifacts["features"]

    # Build combined data: historical + actual 2026 results for prior rounds
    combined_driver_df, _ = build_combined_data(round_num)

    location = race_info["location"]
    is_sprint = 1 if race_info["sprint"] else 0

    # Encode location
    try:
        loc_enc = le_loc.transform([location])[0]
    except ValueError:
        loc_enc = 0

    scores = {}
    for driver, team in GRID_2026.items():
        try:
            d_enc = le_driver.transform([driver])[0]
        except ValueError:
            d_enc = 0

        try:
            t_enc = le_team.transform([team])[0]
        except ValueError:
            t_enc = 0

        # Get lag features from combined data (includes actual 2026 results)
        lags = get_driver_lag_features(combined_driver_df, driver)
        track_avg = get_driver_track_avg(combined_driver_df, driver, location)

        row = {
            "year": year,
            "round": round_num,
            "driver_enc": d_enc,
            "team_enc": t_enc,
            "loc_enc": loc_enc,
            "prev_points": lags["prev_points"],
            "avg_points_3": lags["avg_points_3"],
            "avg_points_5": lags["avg_points_5"],
            "max_points_3": lags["max_points_3"],
            "std_points_3": lags["std_points_3"],
            "yearly_avg_pts": lags["yearly_avg_pts"],
            "yearly_best_pts": lags["yearly_best_pts"],
            "prev_quali": lags["prev_quali"],
            "prev_race_pos": lags["prev_race_pos"],
            "prev_dnf": lags["prev_dnf"],
            "track_avg_pts": track_avg,
            "is_sprint_weekend": is_sprint,
        }

        X = pd.DataFrame([row], columns=features)
        pred = model.predict(X)[0]
        scores[driver] = round(float(pred), 1)

    return scores


def predict_constructor_scores(round_num, year=2026):
    """Predict fantasy points for all 2026 constructors at a given race."""
    race_info = get_race_info(round_num)
    if not race_info:
        return {}

    artifacts = load_constructor_model()
    if not artifacts:
        return {}

    model = artifacts["model"]
    le_team = artifacts["le_team"]
    le_loc = artifacts["le_loc"]
    features = artifacts["features"]

    # Build combined data: historical + actual 2026 results for prior rounds
    _, combined_constructor_df = build_combined_data(round_num)

    location = race_info["location"]
    is_sprint = 1 if race_info["sprint"] else 0

    try:
        loc_enc = le_loc.transform([location])[0]
    except ValueError:
        loc_enc = 0

    teams_2026 = set(GRID_2026.values())
    scores = {}

    for team in teams_2026:
        try:
            t_enc = le_team.transform([team])[0]
        except ValueError:
            t_enc = 0

        lags = get_constructor_lag_features(combined_constructor_df, team)
        track_avg = get_constructor_track_avg(combined_constructor_df, team, location)

        row = {
            "year": year,
            "round": round_num,
            "team_enc": t_enc,
            "loc_enc": loc_enc,
            "c_prev_points": lags["c_prev_points"],
            "c_avg_points_3": lags["c_avg_points_3"],
            "c_avg_points_5": lags["c_avg_points_5"],
            "c_max_points_3": lags["c_max_points_3"],
            "c_std_points_3": lags["c_std_points_3"],
            "c_yearly_avg_pts": lags["c_yearly_avg_pts"],
            "c_yearly_best_pts": lags["c_yearly_best_pts"],
            "c_track_avg": track_avg,
            "is_sprint_weekend": is_sprint,
        }

        X = pd.DataFrame([row], columns=features)
        pred = model.predict(X)[0]
        scores[team] = round(float(pred), 1)

    return scores


def predict_all(round_num, year=2026):
    """Predict both driver and constructor scores for a race."""
    driver_scores = predict_driver_scores(round_num, year)
    constructor_scores = predict_constructor_scores(round_num, year)
    return driver_scores, constructor_scores


def get_actual_round_points(round_num):
    """
    Return actual fantasy points for a specific completed 2026 round.
    Returns (driver_points_dict, constructor_points_dict) or (None, None)
    if the round hasn't been completed yet.
    """
    meetings_2026 = fetch_2026_results()
    if not meetings_2026:
        return None, None

    completed = get_completed_2026_rounds(meetings_2026)
    if round_num not in completed:
        return None, None

    # Filter to just this round's meeting
    round_meetings = [
        m for m in meetings_2026
        if _round_for_location(m["location"]) == round_num
    ]
    if not round_meetings:
        return None, None

    # Process into fantasy point rows
    driver_df, constructor_df = process_raw_data(round_meetings, years_filter=[2026])

    driver_points = {}
    if not driver_df.empty:
        for _, row in driver_df.iterrows():
            driver_points[row["driver"]] = float(row["fantasy_points"])

    constructor_points = {}
    if not constructor_df.empty:
        col = "constructor_fantasy_points"
        for _, row in constructor_df.iterrows():
            constructor_points[row["team"]] = float(row[col])

    return driver_points, constructor_points


def get_all_completed_points():
    """
    Return actual fantasy points for ALL completed 2026 rounds.
    Returns {round_num: (driver_pts_dict, constructor_pts_dict)}.
    """
    meetings_2026 = fetch_2026_results()
    if not meetings_2026:
        return {}

    completed = sorted(get_completed_2026_rounds(meetings_2026))
    results = {}

    for rnd in completed:
        round_meetings = [
            m for m in meetings_2026
            if _round_for_location(m["location"]) == rnd
        ]
        if not round_meetings:
            continue

        driver_df, constructor_df = process_raw_data(round_meetings, years_filter=[2026])

        d_pts = {}
        if not driver_df.empty:
            for _, row in driver_df.iterrows():
                d_pts[row["driver"]] = float(row["fantasy_points"])

        c_pts = {}
        if not constructor_df.empty:
            col = "constructor_fantasy_points"
            for _, row in constructor_df.iterrows():
                c_pts[row["team"]] = float(row[col])

        results[rnd] = (d_pts, c_pts)

    return results


# ──────────────────────────────────────────────────────────────
# Main (test predictions)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  F1 Fantasy 2026 Predictions")
    print("=" * 60)

    for r in [1, 2, 3]:
        race = get_race_info(r)
        if not race:
            continue
        print(f"\n  Round {r}: {race['name']} ({race['location']})")
        print(f"  Sprint weekend: {'Yes' if race['sprint'] else 'No'}")

        d_scores, c_scores = predict_all(r)

        if d_scores:
            print("\n  Driver Predictions:")
            for d, pts in sorted(d_scores.items(), key=lambda x: -x[1]):
                print(f"    {d:15s} → {pts:6.1f} pts")

        if c_scores:
            print("\n  Constructor Predictions:")
            for t, pts in sorted(c_scores.items(), key=lambda x: -x[1]):
                print(f"    {t:20s} → {pts:6.1f} pts")
