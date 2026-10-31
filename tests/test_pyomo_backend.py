"""backend="ipopt": the period as one simultaneous Pyomo system, judged by cgelab's own residual."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

pytest.importorskip("pyomo")
try:
    from cgelab.backends import solver_executable
    solver_executable("ipopt")
except RuntimeError:
    pytest.skip("no ipopt executable", allow_module_level=True)

SHOCK = cge.Scenario(productivity=np.array([1.1, 0.95]), world_price=np.array([1.1, 1.0]))


@pytest.mark.parametrize("closure", [
    cge.Closure(),
    cge.Closure(foreign="fixed_rate", investment="fixed_real"),
    cge.Closure(government_target="revenue", government_adjusts="product_tax"),
], ids=["default", "fixed-rate-real", "revenue-target"])
def test_the_backend_finds_the_scipy_equilibrium(closure):
    _, econ = toy.build()
    ref = cge.solve_period(econ, closure, SHOCK)
    res = cge.solve_period(econ, closure, SHOCK, backend="ipopt")
    assert res.converged and res.attempt == "ipopt" and "IPOPT" in res.message
    assert res.residual < 1e-11
    np.testing.assert_allclose(res.X, ref.X, rtol=1e-10)
    np.testing.assert_allclose(res.factor_prices, ref.factor_prices, rtol=1e-10)
    np.testing.assert_allclose(res.hh_income, ref.hh_income, rtol=1e-10)
    assert abs(cge.national_accounts(res)["gap"]) < 1e-8


def test_the_backend_covers_labour_segments_and_mobility():
    sam, _ = toy.build()
    econ = cge.calibrate(sam, toy.spec(factor_mobility={"f_cap": 1.5}))
    ref = cge.solve_period(econ, scenario=SHOCK)
    res = cge.solve_period(econ, scenario=SHOCK, backend="ipopt")
    np.testing.assert_allclose(res.factor_prices_by_activity, ref.factor_prices_by_activity,
                               rtol=1e-10)


def test_the_backend_is_homogeneous_of_degree_zero():
    _, econ = toy.build()
    one = cge.solve_period(econ, scenario=SHOCK, backend="ipopt")
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0), SHOCK, backend="ipopt")
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10)


def test_the_backend_refuses_what_the_replica_does_not_write():
    _, econ = toy.build()
    with pytest.raises(ValueError, match="backend='ipopt' does not write capacity"):
        cge.solve_period(econ, scenario=cge.Scenario(capacity={"a_ind": 190.0}),
                         backend="ipopt")
    with pytest.raises(ValueError, match="backend must be"):
        cge.solve_period(econ, backend="gams")
    _, regional = regions.build()
    with pytest.raises(ValueError, match="interregional"):
        cge.solve_period(regional, backend="ipopt")
