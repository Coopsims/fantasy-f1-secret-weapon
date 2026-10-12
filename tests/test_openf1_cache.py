import pytest

import fetch_openf1_full as F
import openf1_cache as C
import openf1_utils as U
from f1_weekend_data import build_from_meetings


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, "x"

    def json(self):
        return self._p

    def raise_for_status(self):
        raise AssertionError("not expected")


def _client(monkeypatch, responses):
    seq = iter(responses)
    monkeypatch.setattr(U.requests, "get", lambda *a, **k: next(seq))
    monkeypatch.setattr(U.time, "sleep", lambda s: None)
    c = U.OpenF1Client()
    return c


def test_429_is_retried_slows_the_pacing_and_succeeds(monkeypatch):
    c = _client(monkeypatch, [_Resp(429, headers={"Retry-After": "3"}), _Resp(200, [{"a": 1}])])
    assert c.get_strict("laps", {"meeting_key": 1}) == [{"a": 1}]
    assert c.min_interval >= 0.5


def test_404_is_a_genuine_empty_answer(monkeypatch):
    assert _client(monkeypatch, [_Resp(404)]).get_strict("overtakes", {}) == []


def test_persistent_failure_raises_instead_of_returning_empty(monkeypatch):
    c = _client(monkeypatch, [_Resp(500)] * 3)
    with pytest.raises(U.OpenF1Error):
        c.get_strict("laps", {}, retries=3)
    with pytest.raises(U.OpenF1Error):                       # a bad request is not retried
        _client(monkeypatch, [_Resp(422)]).get_strict("laps", {"x": 1})


def test_pull_status_distinguishes_ok_empty_error(monkeypatch):
    seq = {"a": [1], "b": [], "c": None}

    def fake(ep, params=None):
        if seq[ep] is None:
            raise U.OpenF1Error("boom")
        return seq[ep]
    monkeypatch.setattr(F.openf1_client, "get_strict", fake)
    assert F.pull("a")[1] == "ok" and F.pull("b")[1] == "empty" and F.pull("c")[1].startswith("error")


def _entry():
    drv = lambda sk: [{"session_key": sk, "driver_number": 1, "last_name": "Verstappen", "team_name": "Red Bull Racing"},
                      {"session_key": sk, "driver_number": 63, "last_name": "Russell", "team_name": "Mercedes"}]
    sess = [
        {"session_key": 10, "session_name": "Practice 1", "session_type": "Practice", "date_start": "2026-03-06T01:30:00+00:00"},
        {"session_key": 11, "session_name": "Practice 2", "session_type": "Practice", "date_start": "2026-03-06T05:00:00+00:00"},
        {"session_key": 12, "session_name": "Practice 3", "session_type": "Practice", "date_start": "2026-03-07T04:00:00+00:00"},
        {"session_key": 13, "session_name": "Qualifying", "session_type": "Qualifying", "date_start": "2026-03-07T05:00:00+00:00"},
        {"session_key": 14, "session_name": "Race", "session_type": "Race", "date_start": "2026-03-08T04:00:00+00:00"},
    ]
    pos = lambda sk: [{"session_key": sk, "driver_number": 1, "position": 1, "date": "d"},
                      {"session_key": sk, "driver_number": 63, "position": 2, "date": "d"}]
    lap = lambda sk: [{"session_key": sk, "driver_number": 1, "lap_number": 1, "lap_duration": 80.0,
                       "is_pit_out_lap": False, "date_start": None, "junk": 1}]
    return {"meeting": {"year": 2026, "location": "Melbourne", "meeting_key": 1, "meeting_name": "AUS", "date_start": "2026-03-06"},
            "sessions": sess, "complete": True,
            "endpoints": {"drivers": sum([drv(k) for k in (10, 11, 12, 13, 14)], []),
                          "position": pos(13) + pos(14), "laps": lap(10) + lap(11) + lap(12) + lap(14),
                          "stints": [], "pit": []},
            "status": {}}


def test_legacy_adapter_feeds_the_existing_data_layer():
    legacy = C.to_legacy_meetings([_entry()])
    assert [s["session_type"] for s in legacy[0]["sessions"]] == ["Qualifying", "Race"]
    res = build_from_meetings(legacy)
    assert set(res.sessions["driver"]) == {"Verstappen", "Russell"} and len(res.weekends) == 1


def test_practice_adapter_keeps_only_sessions_before_the_lock():
    p = C.to_practice([_entry()])[0]
    assert p["lock_utc"] == "2026-03-07T05:00:00+00:00"
    assert [s["session_name"] for s in p["sessions"]] == ["Practice 1", "Practice 2", "Practice 3"]
    assert set(p["sessions"][0]["laps"][0]) == set(C.LAP_KEYS)
    sprint = _entry()
    sprint["sessions"] += [{"session_key": 15, "session_name": "Sprint", "session_type": "Race", "date_start": "2026-03-06T08:00:00+00:00"}]
    p2 = C.to_practice([sprint])[0]
    assert p2["lock_utc"] == "2026-03-06T08:00:00+00:00"               # Sprint start beats Qualifying start
    assert [s["session_name"] for s in p2["sessions"]] == ["Practice 1", "Practice 2"]
