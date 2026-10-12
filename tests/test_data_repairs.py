"""Regressions found when the pipeline first met real OpenF1 data."""
from f1_calendar import RACE_CALENDAR_2026, historical_round
from f1_trainer import meeting_gaps
from f1_weekend_data import build_from_meetings


def _pos(order):
    return [{"driver_number": n, "position": i + 1, "date": f"2026-01-01T00:00:{i:02d}"} for i, n in enumerate(order)]


def _meeting(key, location, date, race_lineup, fp1_lineup=None):
    """Q + R meeting. `race_lineup` = {number: (last_name, team)}; the meeting-level list mimics Practice 1."""
    drivers = lambda lu: [{"driver_number": n, "last_name": nm, "team_name": tm} for n, (nm, tm) in lu.items()]  # noqa: E731
    order = list(race_lineup)
    sess = [{"session_key": key * 10 + i, "session_type": t, "drivers": drivers(race_lineup), "positions": _pos(order),
             "laps": [{"driver_number": n, "lap_number": 1, "lap_duration": 90 + n, "date_start": "2026-01-01T00:00:00"}
                      for n in order] if t == "Race" else [],
             "stints": [{"driver_number": n, "lap_end": 50} for n in order] if t == "Race" else [],
             "pit": [{"driver_number": order[0], "lap_number": 20, "stop_duration": 2.4}] if t == "Race" else []}
            for i, t in enumerate(("Qualifying", "Race"))]
    return {"year": 2026, "location": location, "meeting_key": key, "meeting_name": "x", "date_start": date,
            "drivers": drivers(fp1_lineup or race_lineup), "sessions": sess}


RACE = {1: ("Verstappen", "Red Bull Racing"), 63: ("Russell", "Mercedes"), 4: ("Norris", "McLaren")}
FP1 = {1: ("Verstappen", "Red Bull Racing"), 99: ("Reserve", "Mercedes")}      # Russell/Norris absent from FP1


def test_session_lineup_beats_practice_lineup():
    res = build_from_meetings([_meeting(1, "Melbourne", "2026-03-06", RACE, fp1_lineup=FP1)])
    r = res.sessions[res.sessions.session == "R"]
    assert set(r.driver) == {"Verstappen", "Russell", "Norris"}      # nobody lost because FP1 ran a reserve


def test_unscheduled_race_gets_own_round_and_chronological_slot():
    a = _meeting(9, "Kuala Lumpur", "2026-10-02", RACE)          # higher meeting_key but earlier date than Singapore
    b = _meeting(5, "Marina Bay", "2026-10-09", RACE)
    c = _meeting(4, "Baku", "2026-09-24", RACE)
    res = build_from_meetings([b, a, c])
    wk = res.weekends.sort_values("t")
    assert list(wk.location) == ["Baku", "Kuala Lumpur", "Marina Bay"]
    assert wk["round"].is_unique and wk["t"].is_unique
    assert wk["round"].is_monotonic_increasing                         # round order == time order
    assert set(res.sessions.groupby("t").location.nunique()) == {1}   # weekends never merged


def test_round_collision_is_resolved():
    # two meetings that map to the same calendar round must stay separate weekends
    a = _meeting(1, "Madrid", "2026-09-11", RACE)
    b = _meeting(2, "Madrid", "2026-09-12", RACE)
    res = build_from_meetings([a, b])
    assert len(res.weekends) == 2 and res.weekends["round"].is_unique


def test_meeting_gaps_flags_failed_fetches():
    ok = _meeting(1, "Melbourne", "2026-03-06", RACE)
    assert meeting_gaps(ok) == []
    assert meeting_gaps({**ok, "sessions": []}) == ["no sessions"]
    broken = _meeting(1, "Melbourne", "2026-03-06", RACE)
    broken["sessions"][1]["laps"] = []
    assert any("no laps" in g for g in meeting_gaps(broken))
    nodrv = _meeting(1, "Melbourne", "2026-03-06", RACE)
    nodrv["sessions"][0]["drivers"] = []
    assert any("no drivers" in g for g in meeting_gaps(nodrv))


def test_calendar_is_chronological_and_has_kuala_lumpur():
    rounds = [r["round"] for r in RACE_CALENDAR_2026]
    assert rounds == list(range(1, len(rounds) + 1))
    locs = [r["location"] for r in RACE_CALENDAR_2026]
    assert locs.index("Baku") < locs.index("Kuala Lumpur") < locs.index("Marina Bay")
    assert historical_round(2026, "Kuala Lumpur") == locs.index("Kuala Lumpur") + 1


def test_weekend_without_qualifying_or_race_is_skipped():
    m = _meeting(1, "Melbourne", "2026-03-06", RACE)
    m["sessions"] = [m["sessions"][1]]                      # race only: no Q
    ok = _meeting(2, "Shanghai", "2026-03-13", RACE)
    res = build_from_meetings([m, ok])
    assert list(res.weekends.location) == ["Shanghai"]
