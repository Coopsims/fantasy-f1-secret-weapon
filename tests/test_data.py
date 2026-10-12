import numpy as np
import pandas as pd

import f1_rules as R
from f1_calendar import canon_driver, canon_team, historical_round, round_for_location
from f1_weekend_data import (
    build_from_meetings, build_from_training_csv, compute_overtakes, score_weekend_actual,
)


def _iso(sec):
    return (pd.Timestamp("2026-03-08T04:00:00Z") + pd.Timedelta(seconds=sec)).isoformat()


def test_name_canonicalisation():
    assert canon_driver("Hülkenberg") == "Hulkenberg"
    assert canon_driver("Pérez") == "Perez"
    assert canon_driver("Andrea Kimi Antonelli") == "Antonelli"
    assert canon_team("Kick Sauber") == "Audi"
    assert canon_team("Visa Cash App RB") == "Racing Bulls"
    assert canon_team("Haas F1 Team") == "Haas F1 Team"
    assert historical_round(2025, "Montréal") == 10 and historical_round(2024, "Montreal") == 9
    assert round_for_location("Yas Marina") == 25   # Kuala Lumpur inserted at round 18


def test_overtakes_ignore_pit_cycle_swaps():
    # 3 cars, laps start at 0/90/180/270s.  Lap 2: car 3 passes car 2 on track.
    # Lap 3: car 1 pits, so 2 and 3 "gain" on it without overtaking.
    laps = [{"driver_number": d, "lap_number": L, "date_start": _iso(90 * (L - 1))}
            for d in (1, 2, 3) for L in (1, 2, 3, 4)]
    def rec(d, pos, t):
        return {"driver_number": d, "position": pos, "date": _iso(t)}
    positions = [rec(1, 1, -5), rec(2, 2, -5), rec(3, 3, -5),
                 rec(3, 2, 120), rec(2, 3, 120),                    # on-track pass in lap 2
                 rec(1, 3, 250), rec(3, 1, 250), rec(2, 2, 250)]    # lap 3: car 1 in the pits
    pits = [{"driver_number": 1, "lap_number": 3}]
    assert compute_overtakes(positions, laps, pits) == {1: 0, 2: 0, 3: 1}
    # same stream without the pit info would (wrongly) credit passes on the pitter
    no_pit = compute_overtakes(positions, laps, [])
    assert no_pit[3] == 2 and no_pit[2] == 1


def test_overtakes_missing_data_returns_none():
    assert compute_overtakes([], [], []) is None
    assert compute_overtakes([{"driver_number": 1, "position": 1, "date": _iso(0)}], [], []) is None


def _meeting(sprint=False):
    drivers = [
        {"driver_number": 1, "last_name": "Verstappen", "team_name": "Red Bull Racing"},
        {"driver_number": 2, "last_name": "Hülkenberg", "team_name": "Kick Sauber"},
        {"driver_number": 3, "last_name": "Norris", "team_name": "McLaren"},
        {"driver_number": 4, "last_name": "Piastri", "team_name": "McLaren"},
    ]
    def pos(order):
        return [{"driver_number": d, "position": i + 1, "date": _iso(10 + i)} for i, d in enumerate(order)]
    def laps(best):
        return [{"driver_number": d, "lap_number": 1, "lap_duration": t, "date_start": _iso(0)}
                for d, t in best.items()]
    stints = [{"driver_number": d, "lap_end": 50} for d in (1, 3, 4)] + [{"driver_number": 2, "lap_end": 10}]
    sessions = [
        {"session_key": 10, "session_type": "Qualifying", "positions": pos([3, 4, 1, 2]), "laps": [], "stints": [], "pit": []},
        {"session_key": 11, "session_type": "Race", "positions": pos([3, 1, 4, 2]),
         "laps": laps({1: 90.5, 2: 91.0, 3: 90.9, 4: 90.1}), "stints": stints,
         "pit": [{"driver_number": 3, "lap_number": 20, "stop_duration": 2.1},
                 {"driver_number": 1, "lap_number": 22, "stop_duration": 2.6}]},
    ]
    if sprint:
        sessions = [
            {"session_key": 5, "session_type": "Qualifying", "positions": pos([4, 3, 1, 2]), "laps": [], "stints": [], "pit": []},
            {"session_key": 6, "session_type": "Race", "positions": pos([4, 3, 1, 2]),
             "laps": laps({1: 95.0, 2: 95.1, 3: 95.2, 4: 94.9}), "stints": [], "pit": []},
        ] + sessions
    return {"year": 2026, "location": "Melbourne", "meeting_key": 1, "drivers": drivers, "sessions": sessions}


def test_build_from_meetings_basic():
    res = build_from_meetings([_meeting()])
    s = res.sessions
    assert set(s["session"]) == {"Q", "R"}
    assert "Hulkenberg" in set(s["driver"]) and set(s[s.driver == "Hulkenberg"]["team"]) == {"Audi"}
    r = s[s["session"] == "R"].set_index("driver")
    assert r.loc["Hulkenberg", "dnf"] and not r.loc["Norris", "dnf"]
    assert r.loc["Norris", "grid"] == 1 and r.loc["Verstappen", "grid"] == 3
    assert r.loc["Piastri", "fastest_lap"] and r["fastest_lap"].sum() == 1
    assert len(res.pits) == 2 and res.weekends.iloc[0]["sprint"] == False  # noqa: E712


def test_sprint_weekend_relabelled():
    res = build_from_meetings([_meeting(sprint=True)])
    assert set(res.sessions["session"]) == {"Q", "SQ", "S", "R"}
    assert bool(res.weekends.iloc[0]["sprint"])
    sp = res.sessions[res.sessions["session"] == "S"].set_index("driver")
    assert sp.loc["Piastri", "grid"] == 1            # sprint grid comes from sprint qualifying


def test_actual_scoring_matches_hand_calculation():
    res = build_from_meetings([_meeting()])
    d, c = score_weekend_actual(res, 0)
    d = d.set_index("driver")
    # Norris: P1 quali (10) + P1 race (25) from P1 (0 change) = 35, no overtakes (NaN -> 0)
    assert d.loc["Norris", "total"] == 10 + 25
    # Verstappen: P3 quali (8); P2 race (18) + 1 place gained
    assert d.loc["Verstappen", "total"] == 8 + 18 + 1
    # Piastri: P2 quali (9); P3 race (15) - 1 place lost + FL (10)
    assert d.loc["Piastri", "total"] == 9 + 15 - 1 + 10
    # Hulkenberg: no quali points (P4 -> 0), DNF
    assert d.loc["Hulkenberg", "race"] == R.DNF_PENALTY_RACE
    cc = c.set_index("team")
    # McLaren: both in Q3 (+10), quali pts 10+9, race 35-ish... plus pit tier points
    quali_mc = 10 + 9 + R.CONSTRUCTOR_QUALI_Q2_Q3["both_q3"]
    assert cc.loc["McLaren", "quali"] == quali_mc
    assert cc.loc["McLaren", "pit_known"]
    # Norris' 2.1s stop earns the 10 pt tier AND the fastest-stop bonus
    assert cc.loc["McLaren", "race"] == (25 + (15 - 1 + 10)) + 10 + R.PIT_STOP_FASTEST_BONUS
    assert not cc.loc["Audi", "pit_known"]


def test_training_csv_fallback(tmp_path):
    rows = []
    for yr, loc in ((2025, "Melbourne"), (2025, "Shanghai")):
        for i, (drv, team) in enumerate([("Norris", "McLaren"), ("Piastri", "McLaren"), ("Hulkenberg", "Kick Sauber")]):
            rows.append({"year": yr, "location": loc, "driver": drv, "team": team, "fantasy_points": 10,
                         "quali_pos": i + 1, "race_pos": None if i == 2 else i + 1, "dnf": int(i == 2),
                         "overtakes": 0, "fastest_lap": int(i == 0), "is_sprint_weekend": 0, "sprint_pos": None})
    p = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    res = build_from_training_csv(str(p))
    assert len(res.weekends) == 2 and list(res.weekends["round"]) == [1, 2]   # chronological, not alphabetical
    assert set(res.sessions["team"]) == {"McLaren", "Audi"}
    assert res.sessions[(res.sessions.session == "R") & res.sessions.dnf].shape[0] == 2
