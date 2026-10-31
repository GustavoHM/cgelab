"""Capacity limits: limit − output ≥ 0 ⊥ rent ≥ 0, the rent paid to a factor's owners."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def test_a_slack_limit_changes_nothing():
    _, econ = toy.build()
    free = cge.solve_period(econ, scenario=cge.Scenario(productivity=np.array([1.0, 1.1])))
    capped = cge.solve_period(econ, scenario=cge.Scenario(productivity=np.array([1.0, 1.1]),
                                                          capacity={"a_ind": 400.0}))
    np.testing.assert_allclose(capped.X, free.X, rtol=1e-12)
    np.testing.assert_allclose(capped.factor_prices, free.factor_prices, rtol=1e-12)
    assert capped.capacity["a_ind"]["rent"] == 0.0
    assert capped.capacity["a_ind"]["unused"] == pytest.approx(400.0 - free.X[1])


def test_a_binding_limit_holds_output_and_earns_a_rent():
    _, econ = toy.build()
    result = cge.solve_period(econ, scenario=cge.Scenario(capacity={"a_ind": 190.0}))
    assert result.converged and abs(result.walras) < 1e-10
    cap = result.capacity["a_ind"]
    assert result.X[econ.activity("a_ind")] == pytest.approx(190.0, rel=1e-12)
    assert cap["rent"] > 0.01 and cap["unused"] == pytest.approx(0.0, abs=1e-9)
    accounts = cge.national_accounts(result)
    assert abs(accounts["gap"]) < 1e-8
    assert accounts["capacity_rent"] == pytest.approx(cap["rent"] * 190.0)
    # The rent is in the activity's unit cost, so in its commodity's price.
    base = cge.solve_period(econ)
    assert result.pd[econ.commodity("c_ind")] > base.pd[econ.commodity("c_ind")]


def test_a_limit_can_bind_only_after_a_shock():
    _, econ = toy.build()
    limit = {"a_ind": 205.0}
    base = cge.solve_period(econ, scenario=cge.Scenario(capacity=limit))
    assert base.capacity["a_ind"]["rent"] == 0.0
    shocked = cge.solve_period(econ, scenario=cge.Scenario(productivity=np.array([1.0, 1.2]),
                                                           capacity=limit))
    assert shocked.capacity["a_ind"]["rent"] > 0.0
    assert shocked.X[econ.activity("a_ind")] == pytest.approx(205.0, rel=1e-12)


def test_a_capped_cheaper_technology_brings_back_the_dearer_one():
    _, econ = toy.build_with_technologies()
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("a_alt")] = 1.5
    free = cge.solve_period(econ, scenario=cge.Scenario(productivity=tfp))
    assert free.X[econ.activity("a_ind")] == pytest.approx(0.0, abs=1e-10)
    merit = cge.solve_period(econ, scenario=cge.Scenario(productivity=tfp,
                                                         capacity={"a_alt": 200.0}))
    tech = merit.least_cost["c_ind"]
    assert np.all(tech["share"] > 0.1)
    np.testing.assert_allclose(tech["margin"], 0.0, atol=1e-12)
    assert merit.capacity["a_alt"]["rent"] > 0.0
    assert merit.X[econ.activity("a_alt")] == pytest.approx(200.0, rel=1e-12)
    assert abs(cge.national_accounts(merit)["gap"]) < 1e-8


def test_the_rent_goes_to_the_owners_of_the_named_factor():
    _, econ = toy.build()
    limit = {"a_ind": 190.0}
    to_capital = cge.solve_period(econ, scenario=cge.Scenario(capacity=limit))
    to_labour = cge.solve_period(econ, scenario=cge.Scenario(capacity=limit,
                                                             capacity_owner="f_lab"))
    # hh_1 owns three quarters of capital and less of labour; hh_2 the reverse.
    assert to_capital.hh_income[0] > to_labour.hh_income[0]
    assert to_capital.hh_income[1] < to_labour.hh_income[1]
    with pytest.raises(KeyError):
        cge.solve_period(econ, scenario=cge.Scenario(capacity=limit, capacity_owner="f_land"))


def test_capacity_is_homogeneous_of_degree_zero():
    _, econ = toy.build()
    scenario = cge.Scenario(capacity={"a_ind": 190.0})
    one = cge.solve_period(econ, scenario=scenario)
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0), scenario)
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10)
    assert two.capacity["a_ind"]["rent"] == pytest.approx(2.0 * one.capacity["a_ind"]["rent"])


def test_capacity_is_refused_where_it_cannot_hold():
    _, econ = toy.build()
    with pytest.raises(KeyError):
        cge.solve_period(econ, scenario=cge.Scenario(capacity={"a_none": 10.0}))
    with pytest.raises(ValueError, match="nonnegative"):
        cge.solve_period(econ, scenario=cge.Scenario(capacity={"a_ind": -1.0}))
    with pytest.raises(ValueError, match="capacity"):
        cge.to_jump(econ, scenario=cge.Scenario(capacity={"a_ind": 190.0}))
    _, own = toy.build_with_own_production()
    with pytest.raises(ValueError, match="exogenous"):
        cge.solve_period(own, scenario=cge.Scenario(capacity={"a_own": 1.0}))
    _, regional = regions.build()
    with pytest.raises(ValueError, match="capacity"):
        cge.solve_period(regional, scenario=cge.Scenario(capacity={"r1:a": 90.0}))
