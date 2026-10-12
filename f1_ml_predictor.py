"""
F1 Fantasy Predictor (f1_ml_predictor.py)
==========================================
Predicts fantasy points for drivers and constructors at a given 2026 race.

Two back-ends:

  sim   (default)  f1_weekend_model.py - a rating + Monte-Carlo simulation of
                   the whole weekend (qualifying, sprint, race, DNFs, overtakes,
                   fastest lap, pit stops, Driver of the Day).  Gives every event
                   and component separately, with uncertainty.
  xgb              the original gradient-boosted regressors trained by
                   f1_trainer.py (one number per driver per race).
  blend            average of the two.

Choose with the F1_PREDICTOR environment variable, or the `backend=` argument.
From the *second* race onwards both back-ends automatically fetch actual 2026
results from OpenF1 and use them, so every forecast reflects what really
happened in earlier rounds.
"""

import os
import json
import pickle
import time

import pandas as pd
import numpy as np

from f1_calendar import (  # noqa: F401  (re-exported for existing callers)
    GRID_2026, RACE_CALENDAR_2026, get_race_info, LOCATION_ALIASES as _LOCATION_ALIASES,
    round_for_location as _round_for_location,
)

DEFAULT_BACKEND = os.environ.get("F1_PREDICTOR", "sim").lower()
DEFAULT_SIMS = int(os.environ.get("F1_SIMS", "6000"))


def _trainer():
    """Import the (heavy) trainer lazily - only the xgb path / data fetching needs it."""
    import f1_trainer
    return f1_trainer


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

    from f1_trainer import meeting_gaps, backfill_session_drivers
    if backfill_session_drivers(cached):
        with open(CACHE_FILE_2026, "w") as f:
            json.dump(cached, f)
    # a cached meeting with gaps (failed / rate-limited fetch) is retried, not trusted
    cached_keys = {m["meeting_key"] for m in cached if not meeting_gaps(m)}

    from f1_trainer import openf1_get
    from openf1_utils import openf1_client
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
            print(f"  ⚠ {location}: no sessions returned (failed call?) - will retry next run")
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
            "date_start": meeting.get("date_start"),
            "drivers": drivers,
            "sessions": [],
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(get_session_data, s, driver_map): s for s in sessions}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    meeting_data["sessions"].append(result)

        # A meeting is only final once every Race / Sprint session has results.
        # (A future or in-progress race weekend has a scheduled Race session but
        # no positions yet; caching it would freeze it as "empty" forever.)
        race_like = [r for r in meeting_data["sessions"] if r["session_type"] == "Race"]
        if not race_like or any(not r.get("positions") for r in race_like):
            print(f"  ⏳ {location}: race not finished yet - not caching")
            continue

        new_meetings.append(meeting_data)
        time.sleep(0.1)

    if new_meetings:
        replaced = {m["meeting_key"] for m in new_meetings}
        cached = [m for m in cached if m["meeting_key"] not in replaced]
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
    add_round_numbers = _trainer().add_round_numbers
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
    process_raw_data = _trainer().process_raw_data

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

def predict_driver_scores_xgb(round_num, year=2026):
    """XGBoost: predict fantasy points for all 2026 drivers at a given race."""
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


def predict_constructor_scores_xgb(round_num, year=2026):
    """XGBoost: predict fantasy points for all 2026 constructors at a given race."""
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


# ──────────────────────────────────────────────────────────────
# Weekend simulator back-end
# ──────────────────────────────────────────────────────────────

_SIM_STATE = {"forecaster": None, "stamp": None, "cache": {}}
_KNOWN = {}     # round -> real results (quali / sprint_quali) to condition forecasts on


def register_known_results(round_num, known):
    """Remember real session results for a round; later forecasts of it use them."""
    if known:
        _KNOWN[round_num] = known
    else:
        _KNOWN.pop(round_num, None)


def _results_stamp():
    from f1_weekend_data import CURRENT_JSON, HISTORICAL_JSON, DRIVER_CSV
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (CURRENT_JSON, HISTORICAL_JSON, DRIVER_CSV))


def refresh_live_results(round_num):
    """Pull any newly completed 2026 rounds into raw_2026_results.json (best effort)."""
    if round_num <= 1:
        return
    try:
        fetch_2026_results()
    except Exception as exc:   # offline / API down / openf1_utils missing: use what is cached
        print(f"  ℹ Could not refresh live 2026 results ({exc.__class__.__name__}); using cached data.")


def fetch_known_results(round_num, now=None):
    """
    Detect which sessions of this weekend have already finished on OpenF1 and
    return them as a `known` dict for the simulator:

        {"quali": [...], "sprint_quali": [...]}   (driver last names, best first)

    Lets the race forecast be conditioned on the real grid as soon as
    qualifying is over (e.g. before deciding on a Final Fix).  Best effort:
    returns {} when offline or nothing has finished.
    """
    from f1_trainer import openf1_get
    from f1_calendar import canon_driver
    try:
        meetings = openf1_get("meetings", {"year": 2026}) or []
        meeting = next((m for m in meetings if _round_for_location(m.get("location", "")) == round_num), None)
        if not meeting:
            return {}
        sessions = openf1_get("sessions", {"meeting_key": meeting["meeting_key"]}) or []
        drivers = openf1_get("drivers", {"session_key": sessions[0]["session_key"]}) if sessions else []
    except Exception as exc:
        print(f"  ℹ Could not check live sessions ({exc.__class__.__name__}).")
        return {}

    names = {d["driver_number"]: canon_driver(d.get("last_name", "")) for d in drivers}
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    quali = sorted([s for s in sessions if s.get("session_type") == "Qualifying"], key=lambda s: s["session_key"])
    races = [s for s in sessions if s.get("session_type") == "Race"]
    labels = {}
    for i, s in enumerate(quali):
        # sprint weekend: two "Qualifying" sessions, the earlier one is Sprint Qualifying
        labels[s["session_key"]] = "sprint_quali" if (len(quali) == 2 and len(races) == 2 and i == 0) else "quali"

    known = {}
    for s in quali:
        end = pd.to_datetime(s.get("date_end"), utc=True, errors="coerce")
        if pd.isna(end) or end > now:
            continue
        try:
            recs = openf1_get("position", {"session_key": s["session_key"]}) or []
        except Exception:
            continue
        latest = {}
        for r in recs:
            if r.get("driver_number") is not None and r.get("position") is not None:
                latest[r["driver_number"]] = r["position"]
        order = [names[dn] for dn, _ in sorted(latest.items(), key=lambda kv: kv[1]) if dn in names]
        if len(order) >= 10:
            known[labels[s["session_key"]]] = order
    return known


def get_weekend_forecast(round_num, n=None, known=None, exclude=(), grid_penalties=None,
                         seed=7, refresh=True):
    """
    Full probabilistic forecast of a weekend (see f1_weekend_model.WeekendForecast).

    known : real results to condition on, e.g. {"quali": ["Norris", "Piastri", ...]}
            - use after qualifying to re-forecast the race on the true grid.
            Pass "auto" to detect finished sessions on OpenF1.
    """
    from f1_weekend_model import WeekendForecaster
    n = n or DEFAULT_SIMS
    if refresh:
        refresh_live_results(round_num)
    if known == "auto":
        known = fetch_known_results(round_num) or None
    elif known is None:
        known = _KNOWN.get(round_num)
    stamp = _results_stamp()
    if _SIM_STATE["forecaster"] is None or _SIM_STATE["stamp"] != stamp:
        _SIM_STATE.update(forecaster=WeekendForecaster(), stamp=stamp, cache={})
    key = (round_num, n, seed, json.dumps(known, sort_keys=True), tuple(exclude),
           json.dumps(grid_penalties, sort_keys=True))
    cache = _SIM_STATE["cache"]
    if key not in cache:
        cache[key] = _SIM_STATE["forecaster"].forecast(
            round_num, n=n, seed=seed, known=known, exclude=exclude, grid_penalties=grid_penalties)
    return cache[key]


def predict_events(round_num, **kwargs):
    """
    Per-event expected points: {"drivers": {name: {"quali", "sprint", "race", "total", "sd"}},
    "constructors": {...}} - the 'every part of the weekend' view.
    """
    fc = get_weekend_forecast(round_num, **kwargs)
    out = {"drivers": {}, "constructors": {}}
    for _, r in fc.driver_table().iterrows():
        out["drivers"][r["driver"]] = {k: round(float(r[k]), 1) for k in ("quali", "sprint", "race", "total", "sd")}
    for _, r in fc.constructor_table().iterrows():
        out["constructors"][r["team"]] = {k: round(float(r[k]), 1) for k in ("quali", "sprint", "race", "total", "sd")}
    return out


def _resolve_backend(backend):
    return (backend or DEFAULT_BACKEND or "sim").lower()


def _blend(a, b, w=0.5):
    if not a:
        return b
    if not b:
        return a
    return {k: round(w * a[k] + (1 - w) * b.get(k, a[k]), 1) for k in a}


def predict_driver_scores(round_num, year=2026, backend=None):
    """Expected fantasy points for every 2026 driver at `round_num`."""
    backend = _resolve_backend(backend)
    if backend == "xgb":
        return predict_driver_scores_xgb(round_num, year)
    sim = get_weekend_forecast(round_num).driver_scores()
    if backend == "blend":
        return _blend(sim, predict_driver_scores_xgb(round_num, year))
    return sim


def predict_constructor_scores(round_num, year=2026, backend=None):
    """Expected fantasy points for every 2026 constructor at `round_num`."""
    backend = _resolve_backend(backend)
    if backend == "xgb":
        return predict_constructor_scores_xgb(round_num, year)
    sim = get_weekend_forecast(round_num).constructor_scores()
    if backend == "blend":
        return _blend(sim, predict_constructor_scores_xgb(round_num, year))
    return sim


def predict_driver_std(round_num):
    """Simulated standard deviation of each driver's weekend total (for boost choice)."""
    return get_weekend_forecast(round_num).driver_std()


def predict_all(round_num, year=2026, backend=None):
    """Predict both driver and constructor scores for a race."""
    return (predict_driver_scores(round_num, year, backend),
            predict_constructor_scores(round_num, year, backend))


def get_actual_round_points(round_num):
    """
    Return actual fantasy points for a specific completed 2026 round.
    Returns (driver_points_dict, constructor_points_dict) or (None, None)
    if the round hasn't been completed yet.
    """
    process_raw_data = _trainer().process_raw_data
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
    process_raw_data = _trainer().process_raw_data
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
