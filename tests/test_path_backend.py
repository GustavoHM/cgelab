"""backend="path": the period with its complementarity conditions, solved by PATH, judged by cgelab."""

import subprocess

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

pytest.importorskip("pyomo")
try:
    from cgelab.backends import solver_executable
    # The program must run here, not only exist: amplpy ships an x86-64 PATH for macOS.
    subprocess.run([solver_executable("path"), "-v"], capture_output=True, check=True)
except (RuntimeError, OSError, subprocess.CalledProcessError):
    pytest.skip("no runnable PATH executable", allow_module_level=True)


def agree(econ, scenario=None, closure=None):
    ref = cge.solve_period(econ, closure, scenario)
    res = cge.solve_period(econ, closure, scenario, backend="path")
    assert res.converged and res.attempt == "path" and "PATH" in res.message
    assert res.residual < 1e-11
    np.testing.assert_allclose(res.X, ref.X, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(res.factor_prices, ref.factor_prices, rtol=1e-9)
    np.testing.assert_allclose(res.pq, ref.pq, rtol=1e-9)
    np.testing.assert_allclose(res.hh_income, ref.hh_income, rtol=1e-9)
    assert abs(cge.national_accounts(res)["gap"]) < 1e-8
    return res, ref


def technologies(level, **capacity):
    sam, econ = toy.build_with_technologies()
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("a_alt")] = level
    return econ, cge.Scenario(productivity=tfp, capacity=capacity)


def test_without_complementarity_it_finds_the_scipy_equilibrium():
    _, econ = toy.build()
    agree(econ, cge.Scenario(productivity=np.array([1.1, 0.95]),
                             world_price=np.array([1.1, 1.0])))


@pytest.mark.parametrize("level, idle", [(1.2, None), (1.5, "a_ind"), (0.8, "a_alt")])
def test_least_cost_producers_switch_as_in_scipy(level, idle):
    econ, scenario = technologies(level)
    res, _ = agree(econ, scenario)
    if idle is not None:
        assert res.X[econ.activity(idle)] == pytest.approx(0.0, abs=1e-10)


def test_a_binding_capacity_limit_earns_the_scipy_rent():
    econ, scenario = technologies(1.5, a_alt=200.0)
    res, ref = agree(econ, scenario)
    assert res.capacity["a_alt"]["rent"] > 0.0
    assert res.capacity["a_alt"]["rent"] == pytest.approx(ref.capacity["a_alt"]["rent"],
                                                          rel=1e-9)


@pytest.mark.parametrize("holder", ["government", "f_cap"])
def test_a_binding_quota_pays_its_rent_to_its_holder(holder):
    econ, _ = technologies(1.0)
    res, ref = agree(econ, cge.Scenario(import_quota={"c_agr": 13.5}, quota_rent_to=holder))
    assert res.quota["c_agr"]["premium"] == pytest.approx(ref.quota["c_agr"]["premium"],
                                                          rel=1e-9)
    assert res.quota["c_agr"]["premium"] > 0.0


def test_price_bounds_hold_by_subsidy_and_purchases():
    _, econ = toy.build()
    ceiling = cge.Scenario(productivity=np.array([0.8, 1.0]), price_ceiling={"c_agr": 1.1})
    res, ref = agree(econ, ceiling)
    assert res.price_bounds["ceiling"]["c_agr"]["subsidy"] == pytest.approx(
        ref.price_bounds["ceiling"]["c_agr"]["subsidy"], rel=1e-9)
    floor = cge.Scenario(productivity=np.array([1.3, 1.0]), price_floor={"c_agr": 0.871})
    res, ref = agree(econ, floor)
    assert res.stock_quantity[econ.commodity("c_agr")] > 0.0


def test_a_binding_wage_floor_leaves_the_scipy_unemployment():
    _, econ = toy.build(toy.spec(numeraire_factor="f_cap"))
    supply = econ.fac_supply.copy()
    supply[econ.factor("f_cap")] *= 0.8
    res, ref = agree(econ, cge.Scenario(factor_supply=supply),
                     cge.Closure(labour="wage_floor", real_wage=0.95))
    assert res.unemployment > 1.0
    assert res.unemployment == pytest.approx(ref.unemployment, rel=1e-9)


def test_a_backstop_takes_the_market_as_in_scipy():
    sam, _ = toy.build()
    econ = cge.calibrate(sam, toy.spec(),
                         backstops={"a_new": cge.Backstop("c_ind", like="a_ind", markup=1.25)})
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("a_new")] = 1.5
    res, _ = agree(econ, cge.Scenario(productivity=tfp))
    assert res.X[econ.activity("a_ind")] == pytest.approx(0.0, abs=1e-10)


def test_ipopt_still_refuses_complementarity_and_path_refuses_regions():
    econ, scenario = technologies(1.2)
    with pytest.raises(ValueError, match="backend='ipopt' does not write least-cost"):
        cge.solve_period(econ, scenario=scenario, backend="ipopt")
    _, regional = regions.build()
    with pytest.raises(ValueError, match="interregional"):
        cge.solve_period(regional, backend="path")
