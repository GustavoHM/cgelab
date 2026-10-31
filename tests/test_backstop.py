"""Backstop technologies: no base-year output, costs from outside the SAM, entry at least cost."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

MARKUP = 1.25


def economy(**backstops):
    sam, _ = toy.build()
    return cge.calibrate(sam, toy.spec(), backstops=backstops
                         or {"a_new": cge.Backstop("c_ind", like="a_ind", markup=MARKUP)})


def productivity(econ, **levels):
    tfp = np.ones(econ.n_act)
    for name, level in levels.items():
        tfp[econ.activity(name)] = level
    return cge.Scenario(productivity=tfp)


ENGINEERED = cge.Backstop("c_ind", costs={"c_agr": 0.1, "c_ind": 0.2, "f_lab": 0.6, "f_cap": 0.2})


def test_the_base_year_leaves_the_backstop_idle():
    econ = economy()
    assert econ.acts[-1] == "a_new" and econ.X0[-1] == 0.0
    assert econ.least_cost[econ.commodity("c_ind")].tolist() == [1, 2]
    result = cge.solve_period(econ)
    assert result.attempt == "cold" and result.residual < 1e-12
    np.testing.assert_allclose(result.X, [100.0, 200.0, 0.0], atol=1e-10)
    tech = result.least_cost["c_ind"]
    assert tech["share"].tolist() == [1.0, 0.0]
    # It costs the markup on inputs and factors, the activity tax as it is.
    tax = econ.atax[econ.activity("a_ind")]
    assert tech["margin"][1] == pytest.approx(MARKUP * (1.0 - tax) + tax - 1.0)


@pytest.mark.parametrize("level", [1.1, MARKUP])
def test_below_its_markup_the_backstop_stays_idle(level):
    econ = economy()
    _, plain = toy.build()
    result = cge.solve_period(econ, scenario=productivity(econ, a_new=level))
    reference = cge.solve_period(plain)
    np.testing.assert_allclose(result.X[:2], reference.X, rtol=1e-12)
    assert result.X[-1] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("level", [1.3, 1.5, 2.0])
def test_above_its_markup_it_is_the_incumbent_at_productivity_over_markup(level):
    """The known solution: same structure, so the same economy with the incumbent at tfp / m."""
    econ = economy()
    _, plain = toy.build()
    result = cge.solve_period(econ, scenario=productivity(econ, a_new=level))
    reference = cge.solve_period(plain, scenario=productivity(plain, a_ind=level / MARKUP))
    assert result.X[econ.activity("a_ind")] == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(result.X[[0, 2]], reference.X, rtol=1e-12)
    np.testing.assert_allclose(result.factor_prices, reference.factor_prices, rtol=1e-12)
    np.testing.assert_allclose(result.pq, reference.pq, rtol=1e-12)
    np.testing.assert_allclose(result.hh_income, reference.hh_income, rtol=1e-12)
    assert abs(cge.national_accounts(result)["gap"]) < 1e-8


def test_a_backstop_with_another_factor_intensity_can_share_the_market():
    econ = economy(a_eng=ENGINEERED)
    assert econ.io[:, -1] @ (1.0 + econ.ptax) + econ.vunit[-1] == pytest.approx(1.1)
    result = cge.solve_period(econ, scenario=productivity(econ, a_eng=1.11))
    assert result.converged and abs(result.walras) < 1e-10
    tech = result.least_cost["c_ind"]
    assert np.all(tech["share"] > 0.2)
    np.testing.assert_allclose(tech["margin"], 0.0, atol=1e-12)
    assert abs(cge.national_accounts(result)["gap"]) < 1e-8


def test_a_capacity_limit_on_the_incumbent_brings_the_backstop_in():
    econ = economy(a_new=cge.Backstop("c_ind", like="a_ind", markup=1.1))
    scenario = productivity(econ, a_ind=1.2)
    free = cge.solve_period(econ, scenario=scenario)
    assert free.X[-1] == pytest.approx(0.0, abs=1e-10)
    capped = cge.solve_period(econ, scenario=scenario.with_(capacity={"a_ind": 210.0}))
    assert capped.X[econ.activity("a_ind")] == pytest.approx(210.0, rel=1e-12)
    assert capped.X[-1] > 1.0 and capped.capacity["a_ind"]["rent"] > 0.0
    np.testing.assert_allclose(capped.least_cost["c_ind"]["margin"][1], 0.0, atol=1e-12)


def test_a_backstop_joins_two_technologies_already_competing():
    sam, _ = toy.build_with_technologies()
    econ = cge.calibrate(sam, toy.spec(least_cost=["c_ind"]),
                         backstops={"a_new": cge.Backstop("c_ind", like="a_alt", markup=1.2)})
    assert econ.least_cost[econ.commodity("c_ind")].size == 3
    base = cge.solve_period(econ)
    assert base.residual < 1e-12 and base.X[-1] == 0.0
    result = cge.solve_period(econ, scenario=productivity(econ, a_new=1.5))
    assert result.converged and result.X[-1] > 1.0


def test_a_backstop_like_an_activity_emits_as_it_does():
    econ = cge.add_emissions(economy(a_new=cge.Backstop("c_ind", like="a_ind", markup=MARKUP),
                                     a_eng=ENGINEERED),
                             process={"a_ind": 20.0})
    process = econ.emissions.process
    assert process[econ.activity("a_new")] == process[econ.activity("a_ind")] > 0.0
    assert process[econ.activity("a_eng")] == 0.0


def test_a_backstop_is_homogeneous_of_degree_zero():
    econ = economy(a_eng=ENGINEERED)
    scenario = productivity(econ, a_eng=1.11)
    one = cge.solve_period(econ, scenario=scenario)
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0), scenario)
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10)


@pytest.mark.parametrize("backstop, error, match", [
    (cge.Backstop("c_none", like="a_ind", markup=1.2), KeyError, "c_none"),
    (cge.Backstop("c_ind"), ValueError, "either like"),
    (cge.Backstop("c_ind", like="a_ind", markup=1.2, costs={"f_lab": 2.0}), ValueError, "either like"),
    (cge.Backstop("c_ind", like="a_none", markup=1.2), KeyError, "a_none"),
    (cge.Backstop("c_ind", like="a_ind"), ValueError, "markup"),
    (cge.Backstop("c_ind", like="a_ind", markup=1.0), ValueError, "base year"),
    (cge.Backstop("c_ind", costs={"f_lab": 0.5, "f_cap": 0.5}), ValueError, "base year"),
    (cge.Backstop("c_ind", costs={"f_land": 2.0}), KeyError, "f_land"),
    (cge.Backstop("c_ind", costs={"f_lab": -1.0, "f_cap": 3.0}), ValueError, "nonnegative"),
    (cge.Backstop("c_ind", costs={"f_lab": 2.0}, markup=1.2), ValueError, "markup"),
], ids=["commodity", "neither", "both", "like", "no-markup", "markup-1", "too-cheap",
        "account", "negative", "costs-markup"])
def test_a_backstop_is_refused_when_it_cannot_be_calibrated(backstop, error, match):
    with pytest.raises(error, match=match):
        economy(a_new=backstop)


def test_a_backstop_is_refused_where_it_cannot_hold():
    with pytest.raises(ValueError, match="already the name"):
        economy(a_ind=cge.Backstop("c_ind", like="a_ind", markup=1.2))
    sam, _ = toy.build_with_own_production()
    with pytest.raises(ValueError, match="own-account"):
        cge.calibrate(sam, toy.spec(), backstops={"a_new": cge.Backstop("c_ind", like="a_own",
                                                                        markup=1.2)})
    sam, _ = toy.build()
    with pytest.raises(ValueError, match="energy nest"):
        cge.calibrate(sam, toy.spec(energy=["c_ind"]), backstops={"a_new": ENGINEERED})
    with pytest.raises(ValueError, match="imperfectly"):
        cge.calibrate(sam, toy.spec(factor_mobility={"f_cap": 1.0}),
                      backstops={"a_new": ENGINEERED})
    mobile = cge.calibrate(sam, toy.spec(factor_mobility={"f_cap": 1.0}),
                           backstops={"a_new": cge.Backstop("c_ind", costs={"f_lab": 1.2})})
    assert cge.solve_period(mobile).residual < 1e-12
    rsam, _ = regions.build()
    with pytest.raises(ValueError, match="national"):
        cge.calibrate(rsam, regions.spec(), backstops={"a_new": ENGINEERED})
    with pytest.raises(ValueError, match="least-cost"):
        cge.to_jump(economy())
