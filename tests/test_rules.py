import numpy as np
import f1_rules as R


def test_quali_and_finish_tables():
    assert R.quali_points(1) == 10 and R.quali_points(10) == 1 and R.quali_points(11) == 0
    assert R.race_finish_points(1) == 25 and R.race_finish_points(10) == 1 and R.race_finish_points(15) == 0
    assert R.sprint_finish_points(8) == 1 and R.sprint_finish_points(9) == 0


def test_position_change_symmetry():
    assert R.position_change_points(10, 4) == 6
    assert R.position_change_points(3, 8) == -5
    assert R.position_change_points(5, 5) == 0


def test_race_scoring_components():
    c = R.score_race_like("race", grid=5, finish=2, dnf=False, overtakes=4, fastest_lap=True)
    assert c["finish"] == 18 and c["position_change"] == 3
    assert c["overtakes"] == 4 and c["fastest_lap"] == 10 and c["dnf"] == 0
    assert R.event_total(c) == 18 + 3 + 4 + 10


def test_dnf_loses_finish_and_position_points_but_keeps_overtakes():
    c = R.score_race_like("race", grid=2, finish=22, dnf=True, overtakes=3, fastest_lap=False)
    assert R.event_total(c) == -20 + 3
    s = R.score_race_like("sprint", 2, 22, True, 0, False)
    assert R.event_total(s) == -10


def test_vectorised_matches_scalar():
    grid = np.array([[1, 5], [9, 3]])
    fin = np.array([[2, 1], [10, 12]])
    z = np.zeros_like(grid)
    tot = R.event_total(R.score_race_like("race", grid, fin, z.astype(bool), z, z.astype(bool)))
    for i in range(2):
        for j in range(2):
            one = R.event_total(R.score_race_like("race", grid[i, j], fin[i, j], False, 0, False))
            assert tot[i, j] == one


def test_constructor_quali_bonus_tiers():
    f = R.constructor_quali_bonus
    assert f(1, 10) == 10                    # both in Q3
    assert f(1, 11) == 5                     # one in Q3
    assert f(11, 16) == 3                    # both reach Q2 (16 advance from Q1)
    assert f(11, 17) == 1                    # one reaches Q2
    assert f(17, 22) == -1                   # neither
    assert f(3, 5, nc_a=True) == 5           # NC driver counts as missing Q2/Q3


def test_pit_stop_tiers():
    t = np.array([1.9, 2.1, 2.3, 2.7, 3.2, np.nan, 1.7])
    assert list(R.pit_stop_points(t)) == [20, 10, 5, 2, 0, 0, 20 + 15]


def test_nc_penalty():
    assert R.score_qualifying(3, nc=True) == R.QUALI_NC_PENALTY
