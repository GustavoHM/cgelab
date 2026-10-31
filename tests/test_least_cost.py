"""Producers that compete at least cost: unit cost ≥ price ⊥ output share ≥ 0."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def _alt(econ, value):
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("a_alt")] = value
    return cge.Scenario(productivity=tfp)


def test_the_base_year_has_every_producer_active_at_one_price():
    sam, econ = toy.build_with_technologies()
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-10
    np.testing.assert_allclose(result.X, econ.X0, rtol=1e-12)
    tech = result.least_cost["c_ind"]
    assert tech["activities"] == ["a_alt", "a_ind"]
    np.testing.assert_allclose(tech["share"], [0.4, 0.6], atol=1e-12)
    np.testing.assert_allclose(tech["margin"], 0.0, atol=1e-12)


@pytest.mark.parametrize("value, off", [(1.5, "a_ind"), (0.8, "a_alt")])
def test_the_costlier_technology_shuts_down(value, off):
    _, econ = toy.build_with_technologies()
    result = cge.solve_period(econ, scenario=_alt(econ, value))
    assert result.converged and abs(result.walras) < 1e-10
    tech = result.least_cost["c_ind"]
    k = tech["activities"].index(off)
    assert result.X[econ.activity(off)] == pytest.approx(0.0, abs=1e-10)
    assert tech["share"][k] == pytest.approx(0.0, abs=1e-12)
    assert tech["margin"][k] > 1e-4                       # it would lose money
    assert tech["margin"][1 - k] == pytest.approx(0.0, abs=1e-12)
    # The commodity's output is all the active producer's.
    j = econ.activity(tech["activities"][1 - k])
    assert result.X[j] == pytest.approx(result.Q[econ.commodity("c_ind")], rel=1e-12)


def test_between_the_switches_both_produce_at_one_price():
    _, econ = toy.build_with_technologies()
    result = cge.solve_period(econ, scenario=_alt(econ, 1.2))
    tech = result.least_cost["c_ind"]
    assert np.all(tech["share"] > 0.1)
    np.testing.assert_allclose(tech["unit_cost"], tech["price"], rtol=1e-12)
    assert tech["price"] == pytest.approx(result.pd[econ.commodity("c_ind")], rel=1e-14)


def test_an_idle_technology_does_not_matter_how_far_behind_it_is():
    _, econ = toy.build_with_technologies()
    near, far = (cge.solve_period(econ, scenario=_alt(econ, v)) for v in (0.8, 0.5))
    np.testing.assert_allclose(near.X, far.X, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(near.factor_prices, far.factor_prices, rtol=1e-10)
    assert far.least_cost["c_ind"]["margin"][0] > near.least_cost["c_ind"]["margin"][0]


def _one_factor():
    """Every factor payment on labour: unit costs then rank the technologies alone."""
    entries = [("f_lab" if r == "f_cap" else r, "f_lab" if c == "f_cap" else c, v)
               for r, c, v in [e for e in toy._BASE if "a_ind" not in e[:2]] + toy.TECHNOLOGIES]
    sam = cge.SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]))
    return cge.calibrate(sam, toy.spec(factors=("f_lab",), least_cost=["c_ind"]))


@pytest.mark.parametrize("value, off", [(1.001, "a_ind"), (0.999, "a_alt")])
def test_with_one_factor_any_cost_advantage_takes_the_whole_market(value, off):
    econ = _one_factor()
    result = cge.solve_period(econ, scenario=_alt(econ, value))
    assert result.converged and abs(result.walras) < 1e-10
    tech = result.least_cost["c_ind"]
    assert tech["share"][tech["activities"].index(off)] == pytest.approx(0.0, abs=1e-12)


def test_least_cost_is_homogeneous_of_degree_zero():
    _, econ = toy.build_with_technologies()
    one = cge.solve_period(econ, scenario=_alt(econ, 1.5))
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0), _alt(econ, 1.5))
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(two.pd, 2.0 * one.pd, rtol=1e-10)
    np.testing.assert_allclose(two.least_cost["c_ind"]["margin"],
                               2.0 * one.least_cost["c_ind"]["margin"], rtol=1e-8, atol=1e-12)


def test_least_cost_is_refused_where_it_cannot_hold():
    sam, _ = toy.build_with_technologies()
    with pytest.raises(KeyError, match="c_none"):
        cge.calibrate(sam, toy.spec(least_cost=["c_none"]))
    with pytest.raises(ValueError, match="no other technology"):
        cge.calibrate(sam, toy.spec(least_cost=["c_agr"]))
    with pytest.raises(ValueError, match="repeats"):
        cge.calibrate(sam, toy.spec(least_cost=["c_ind", "c_ind"]))
    with pytest.raises(ValueError, match="national"):
        regions.build(regions.spec(least_cost=["r1:c"]))
    _, econ = toy.build_with_technologies()
    with pytest.raises(ValueError, match="least-cost"):
        cge.to_jump(econ)


def test_a_producer_of_two_commodities_cannot_compete_at_least_cost():
    sam, _ = toy.build_with_technologies()
    frame = pd.concat([sam.frame, pd.DataFrame(
        [("a_alt", "c_agr", 5.0), ("c_agr", "a_alt", 5.0)], columns=["row", "col", "value"])])
    with pytest.raises(ValueError, match="makes"):
        cge.calibrate(cge.SAM.from_long(frame), toy.spec(least_cost=["c_ind"]))
