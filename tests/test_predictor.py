import sys
import types

import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

if "openf1_utils" not in sys.modules:
    try:
        import openf1_utils  # noqa: F401
    except ImportError:      # repo doesn't ship it; stub so the trainer can be imported
        stub = types.ModuleType("openf1_utils")
        stub.openf1_client = types.SimpleNamespace(token=None, check_refresh=lambda: None,
                                                   get=lambda *a, **k: [])
        sys.modules["openf1_utils"] = stub

import f1_ml_predictor as P  # noqa: E402
import f1_trainer as T       # noqa: E402
from f1_synthetic import synthetic_results  # noqa: E402
from f1_weekend_model import WeekendForecaster  # noqa: E402


@pytest.fixture(autouse=True)
def _synthetic_forecaster(monkeypatch):
    res = synthetic_results(seed=2, rounds_2026=8)
    P._SIM_STATE.update(forecaster=WeekendForecaster(res, verbose=False), stamp=P._results_stamp(), cache={})
    monkeypatch.setattr(P, "refresh_live_results", lambda r: None)
    yield
    P._SIM_STATE.update(forecaster=None, stamp=None, cache={})


def test_default_backend_is_simulator():
    d = P.predict_driver_scores(9)
    c = P.predict_constructor_scores(9)
    assert set(d) == set(T.GRID_2026) and len(c) == 11
    assert all(isinstance(v, float) for v in d.values())


def test_one_simulation_serves_driver_and_constructor_calls():
    P.predict_driver_scores(9)
    n_cached = len(P._SIM_STATE["cache"])
    P.predict_constructor_scores(9)
    P.predict_driver_std(9)
    assert len(P._SIM_STATE["cache"]) == n_cached == 1


def test_predict_events_has_every_part_of_the_weekend():
    ev = P.predict_events(14)           # Zandvoort: sprint weekend
    nor = ev["drivers"]["Norris"]
    assert {"quali", "sprint", "race", "total", "sd"} <= set(nor)
    assert abs(nor["quali"] + nor["sprint"] + nor["race"] - nor["total"]) < 0.3
    assert ev["constructors"]["McLaren"]["sprint"] != 0


def test_known_results_condition_the_forecast():
    order = list(reversed(list(T.GRID_2026)))
    fc = P.get_weekend_forecast(9, known={"quali": order})
    assert fc.driver_table().set_index("driver").loc[order[0], "P_pole"] == 1.0


def test_trainer_reexports_shared_rules():
    import f1_rules
    assert T.RACE_PTS is f1_rules.RACE_PTS and T.GRID_2026 is P.GRID_2026


def test_trainer_round_numbers_are_chronological():
    df = pd.DataFrame({"year": [2025] * 3, "location": ["Melbourne", "Zandvoort", "Austin"]})
    assert list(T.add_round_numbers(df)["round"]) == [1, 2, 3]       # not alphabetical


def test_fetch_known_results(monkeypatch):
    drivers = [{"driver_number": i + 1, "last_name": n}
               for i, n in enumerate(["Norris", "Piastri", "Hülkenberg"] + [f"X{i}" for i in range(9)])]
    data = {
        "meetings": [{"meeting_key": 7, "location": "Melbourne"}],
        "sessions": [
            {"session_key": 1, "session_type": "Qualifying", "date_end": "2026-03-07T06:00:00+00:00"},
            {"session_key": 2, "session_type": "Race", "date_end": "2026-03-08T06:00:00+00:00"},
        ],
        "drivers": drivers,
        "position": [{"driver_number": dn, "position": 13 - dn} for dn in range(1, 13)],
    }
    monkeypatch.setattr(T, "openf1_get", lambda ep, params=None, retries=3: data[ep])
    known = P.fetch_known_results(1, now="2026-03-07T12:00:00Z")
    assert known["quali"][0] == "X8" and known["quali"][-1] == "Norris"      # position 1 = driver 12
    assert "Hulkenberg" in known["quali"]
    # before the session ends nothing is known
    assert P.fetch_known_results(1, now="2026-03-07T05:00:00Z") == {}


def test_incomplete_meetings_are_not_cached(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = {
        "meetings": [{"meeting_key": 3, "location": "Melbourne", "meeting_name": "AUS"}],
        "sessions": [{"session_key": 30, "session_type": "Qualifying"}, {"session_key": 31, "session_type": "Race"}],
        "drivers": [{"driver_number": 1, "last_name": "Norris", "team_name": "McLaren"}],
        "position": [],
        "laps": [], "stints": [], "pit": [],
    }
    monkeypatch.setattr(T, "openf1_get", lambda ep, params=None, retries=3: calls[ep])
    out = P.fetch_2026_results()
    assert out == [] and not (tmp_path / P.CACHE_FILE_2026).exists()


def test_trainer_and_simulator_score_identically():
    """The legacy per-driver scoring and the new shared rules must agree."""
    from tests.test_data import _meeting
    from f1_weekend_data import build_from_meetings, score_weekend_actual
    for sprint in (False, True):
        m = _meeting(sprint=sprint)
        d_old, _ = T.process_raw_data([m])
        d_new, _ = score_weekend_actual(build_from_meetings([m]), 0)
        old = d_old.set_index("driver")["fantasy_points"]
        new = d_new.set_index("driver")["total"]
        assert (old - new.reindex(old.index)).abs().max() < 1e-9, (sprint, old, new)
