import numpy as np
import pandas as pd

from f1_practice import build_practice_features, _session_table


def _laps(driver, durations, start=1, out_laps=()):
    return [{"driver_number": driver, "lap_number": start + i, "lap_duration": d,
             "is_pit_out_lap": (start + i) in out_laps, "date_start": None} for i, d in enumerate(durations)]


def _session(laps, stints):
    drivers = [{"driver_number": 1, "last_name": "Verstappen"}, {"driver_number": 63, "last_name": "Russell"},
               {"driver_number": 4, "last_name": "Norris"}]
    return {"session_key": 1, "session_name": "Practice 2", "date_start": "x", "drivers": drivers,
            "laps": laps, "stints": stints}


def test_session_table_gap_inputs_ignore_out_laps_and_slow_laps():
    laps = _laps(1, [90.0, 80.5, 80.2, 80.9, 81.0, 80.7, 80.6], out_laps=(1,)) \
        + _laps(63, [99.0, 81.0, 81.1, 85.0]) + _laps(4, [60.0, 120.0])      # 60.0 is a lap on lap 1: out-lap -> ignored below
    laps[-2]["is_pit_out_lap"] = True
    t = _session_table(_session(laps, [{"driver_number": 1, "lap_start": 1, "lap_end": 7}])).set_index("driver")
    assert t.loc["Verstappen", "best"] == 80.2 and t.loc["Russell", "best"] == 81.0
    assert t.loc["Verstappen", "lr"] > 0 and np.isnan(t.loc["Russell", "lr"])        # only a >=5-lap stint counts
    assert "Norris" not in t.index                                                   # 120s lap is outside 107%, 60s is an out-lap


def test_features_only_use_sessions_given_and_team_comparisons():
    laps = _laps(1, [80.0, 80.2]) + _laps(63, [80.8, 80.9]) + _laps(4, [80.4, 80.6])
    pdata = [{"meeting_key": 1, "year": 2025, "location": "Melbourne", "lock_utc": "x", "sessions": [_session(laps, [])]}]
    weekends = pd.DataFrame({"t": [7], "year": [2025], "location": ["Melbourne"]})
    sessions = pd.DataFrame({"t": [7, 7, 7, 7], "session": ["R", "R", "R", "SQ"],
                             "driver": ["Verstappen", "Russell", "Norris", "Norris"],
                             "team": ["Red Bull Racing", "Mercedes", "McLaren", "McLaren"], "finish": [1, 2, 3, 1]})
    f = build_practice_features(pdata, weekends, sessions).set_index("driver")
    assert f.loc["Verstappen", "fp_gap_min"] == 0 and abs(f.loc["Russell", "fp_gap_min"] - 1.0) < 1e-9
    assert f.loc["Verstappen", "fp_rank_mean"] == 1 and f.loc["Norris", "fp_rank_mean"] == 2
    assert f.loc["Norris", "sq_pos"] == 1 and np.isnan(f.loc["Russell", "sq_pos"])
    assert f.loc["Verstappen", "fp_team_gap_min"] == f.loc["Verstappen", "fp_gap_min"]
