import numpy as np

import f1_planner as FP
from final_fix import final_fix_options


def test_final_fix_ranks_affordable_swaps_and_doubles_a_boosted_swap():
    team = FP.DRIVERS[:5] + FP.TEAMS[:2]
    price = np.full(len(FP.ENTITIES), 10.0)
    rich, cheap = FP.DRIVERS[5], FP.DRIVERS[6]
    price[FP.ENTITIES.index(rich)] = 30.0                    # not affordable with bank 1
    draws = {e: np.zeros(100) for e in FP.ENTITIES}
    draws[rich] = np.full(100, 50.0)
    draws[cheap] = np.full(100, 8.0)
    draws[team[0]] = np.full(100, 3.0)                     # the boosted driver
    opts = final_fix_options(team, team[0], 1.0, price, draws)
    assert rich not in set(opts["in"])
    best = opts.iloc[0]
    assert best["out"] == team[0] and best["in"] == cheap and best["boosted"]
    assert abs(best["exp_gain"] - 2 * (8.0 - 3.0)) < 1e-9 and best["p_better"] == 1.0
    assert set(opts["out"]) <= set(FP.DRIVERS)                # constructors only with allow_constructors
