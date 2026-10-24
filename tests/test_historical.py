"""Historical calibration against hand roots and known simulated histories."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.historical import _scalar_root, _vector_root


class Solved:
    converged = True

    def __init__(self, x):
        self.x = x


def hand_solve(x, **kwargs):
    return Solved(x), None


def test_scalar_and_joint_roots_match_log_solutions_written_by_hand():
    x = _scalar_root(hand_solve, lambda res: 3 * np.exp(res.x), 9.0, 0.0, 2021, 1e-12)
    assert x == pytest.approx(np.log(3), rel=1e-12)

    def measured(res):
        return np.array([2 * np.exp(res.x[0]), 5 * np.exp(res.x[1])])

    x = _vector_root(hand_solve, measured, np.array([4.0, 15.0]),
                     np.zeros(2), 2021, 1e-12, ["x", "y"])
    np.testing.assert_allclose(x, np.log([2.0, 3.0]), rtol=1e-10)


def gdp(result):
    return cge.real_gdp(result)["expenditure"]


def test_productivity_recovers_a_known_history_and_replays():
    _, econ = toy.build()
    known = np.full(econ.n_act, 1.1)
    observed = cge.solve_period(econ, scenario=cge.Scenario(productivity=known))
    hist = cge.calibrate_historical(
        econ, years=[2021], targets={"gdp": (pd.Series({2021: gdp(observed)}), gdp)},
        shifters={"gdp": ("productivity",)})
    np.testing.assert_allclose(hist.multiplier.loc[2021], known, rtol=1e-9)
    assert hist.achieved.loc[2021, "gdp"] == pytest.approx(gdp(observed), rel=1e-10)
    assert gdp(hist.path[2021]) == pytest.approx(gdp(hist.run()[2021]), rel=1e-10)


def preferred_economy(econ, multiplier):
    budget = econ.hh_budget.copy()
    weights = budget[0] * np.array([multiplier, 1.0])
    budget[0] = weights / weights.sum()
    return dataclasses.replace(econ, hh_budget=budget)


def test_preferences_are_relative_normalised_and_recoverable():
    _, econ = toy.build()
    observed_econ = preferred_economy(econ, 1.25)
    observed = cge.solve_period(observed_econ)
    target = observed.hh_quantity[0, 0]
    hist = cge.calibrate_historical(
        econ, years=[2021],
        targets={"food": (pd.Series({2021: target}), lambda r: r.hh_quantity[0, 0])},
        shifters={"food": ("preference", "hh_1", "c_agr")},
        preference_reference={"hh_1": "c_ind"})
    assert hist.preferences["hh_1"].loc[2021, "c_agr"] == pytest.approx(1.25, rel=1e-8)
    assert hist.preferences["hh_1"].loc[2021, "c_ind"] == 1.0
    assert hist.path[2021].econ.hh_budget[0].sum() == pytest.approx(1.0)
    assert hist.path[2021].hh_quantity[0, 0] == pytest.approx(target, rel=1e-9)
    assert hist.run()[2021].hh_quantity[0, 0] == pytest.approx(target, rel=1e-9)


def test_productivity_and_preferences_are_solved_jointly():
    _, econ = toy.build()
    productivity = np.full(econ.n_act, 1.05)
    observed_econ = preferred_economy(econ, 1.2)
    observed = cge.solve_period(observed_econ, scenario=cge.Scenario(productivity=productivity))
    hist = cge.calibrate_historical(
        econ, years=[2021],
        targets={"gdp": (pd.Series({2021: gdp(observed)}), gdp),
                 "food": (pd.Series({2021: observed.hh_quantity[0, 0]}),
                          lambda r: r.hh_quantity[0, 0])},
        shifters={"gdp": ("productivity",),
                  "food": ("preference", "hh_1", "c_agr")},
        preference_reference={"hh_1": "c_ind"})
    np.testing.assert_allclose(hist.multiplier.loc[2021], productivity, rtol=1e-8)
    assert hist.preferences["hh_1"].loc[2021, "c_agr"] == pytest.approx(1.2, rel=1e-7)
    np.testing.assert_allclose(hist.achieved.loc[2021], hist.target.loc[2021], rtol=1e-8)


def test_historical_refuses_unidentified_or_invalid_specifications():
    _, econ = toy.build()
    values = pd.Series({2021: 100.0})
    with pytest.raises(ValueError, match="same entries"):
        cge.calibrate_historical(econ, years=[2021], targets={"x": (values, gdp)}, shifters={})
    with pytest.raises(ValueError, match="distinct"):
        cge.calibrate_historical(
            econ, years=[2021], targets={"x": (values, gdp), "y": (values, gdp)},
            shifters={"x": ("productivity",), "y": ("productivity",)})
    with pytest.raises(ValueError, match="preference_reference"):
        cge.calibrate_historical(
            econ, years=[2021], targets={"x": (values, lambda r: r.hh_quantity[0, 0])},
            shifters={"x": ("preference", "hh_1", "c_agr")})
    with pytest.raises(ValueError, match="reference commodity"):
        cge.calibrate_historical(
            econ, years=[2021], targets={"x": (values, lambda r: r.hh_quantity[0, 0])},
            shifters={"x": ("preference", "hh_1", "c_agr")},
            preference_reference={"hh_1": "c_agr"})
    with pytest.raises(ValueError, match="positive"):
        cge.calibrate_historical(
            econ, years=[2021], targets={"x": (pd.Series({2021: 0.0}), gdp)},
            shifters={"x": ("productivity",)})
