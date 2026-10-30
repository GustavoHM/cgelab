"""The exact Jacobian of the regional residual against central finite differences."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
import cgelab.regional as regional
from cgelab import regional_jacobian
from cgelab.examples import regions
from tests.test_regional_government import labelled_institutions
from tests.test_regional_institutions import build_institutional


@pytest.fixture
def captured(monkeypatch):
    """The residual and Jacobian functions solve_regions hands to its root finder."""
    seen = {}
    original = regional._solve

    def spy(residuals, n_unknown, tolerance, warm_start, raise_on_fail, jac=None):
        seen.update(residuals=residuals, jac=jac)
        return original(residuals, n_unknown, tolerance, warm_start, raise_on_fail, jac=jac)

    monkeypatch.setattr(regional, "_solve", spy)
    return seen


def _central(f, z, h=1e-6):
    cols = []
    for k in range(z.size):
        step = np.zeros(z.size)
        step[k] = h
        cols.append((f(z + step) - f(z - step)) / (2 * h))
    return np.array(cols).T


def _migration(form):
    proximity = pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], index=["r1", "r2"],
                             columns=["r1", "r2"])
    return cge.calibrate(regions.build()[0],
                         regions.spec(factor_migration={"f_lab": 2.0}, factor_mobility_form=form),
                         factor_pools={"f_lab": proximity})


TFP = np.array([1.0, 1.1, 1.0, 1.0])
CASES = {
    "symmetric": (lambda: regions.build()[1], cge.Closure(),
                  cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.2]))),
    "asymmetric_cet": (lambda: regions.build(regions.spec(export_rule="cet", sigma_cet=3.0),
                                             asymmetric_imports=True)[1], cge.Closure(),
                       cge.Scenario(export_price=np.array([1.3]), world_price=np.array([0.9]))),
    "transport_intermediates": (lambda: regions.build_with_transport(intermediates=True)[1],
                                cge.Closure(), cge.Scenario(productivity=TFP)),
    "transport_regional_investment": (
        lambda: regions.build_with_transport(regional_investment=True)[1], cge.Closure(),
        cge.Scenario(productivity=TFP)),
    "transport_fixed_investment": (
        lambda: regions.build_with_transport(regional_investment=True)[1],
        cge.Closure(investment="fixed_real"), cge.Scenario(productivity=TFP, investment_level=50.0)),
    "service_exports_cet": (
        lambda: regions.build_with_transport(service_exports=True, export_rule="cet",
                                             sigma_cet=3.0)[1], cge.Closure(),
        cge.Scenario(productivity=TFP, export_price=np.array([1.0, 1.3]))),
    "migration_cet": (lambda: _migration("cet"), cge.Closure(),
                      cge.Scenario(productivity=np.array([0.8, 1.0]))),
    "migration_logit": (lambda: _migration("logit"), cge.Closure(),
                        cge.Scenario(productivity=np.array([0.8, 1.0]))),
    "institutions_nominal": (build_institutional, cge.Closure(government="nominal_fixed"),
                             cge.Scenario(productivity=np.array([1.1, 1.0]))),
    "taxes_transfers": (lambda: labelled_institutions()[1], cge.Closure(investment="fixed_real"),
                        cge.Scenario(world_price=np.array([1.0, 1.5]), investment_level=60.0)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_exact_jacobian_matches_finite_differences(captured, case):
    build, closure, scenario = CASES[case]
    result = cge.solve_period(build(), closure, scenario)
    f, jac = captured["residuals"], captured["jac"]
    assert jac is not None
    for z in (result.z, result.z + np.linspace(0.02, -0.03, result.z.size)):
        exact, numeric = jac(z), _central(f, z)
        assert exact.shape == numeric.shape
        # Central differences with this step are exact to about 1e-10.
        assert np.max(np.abs(exact - numeric)) < 2e-9 * max(np.max(np.abs(numeric)), 1.0)


def test_closures_outside_the_derivation_keep_finite_differences(captured):
    econ = build_institutional()
    result = cge.solve_period(econ, cge.Closure(government="saving_rate"))
    assert captured["jac"] is None and result.converged
    assert not regional_jacobian.supported(
        cge.Closure(government_target="revenue", government_adjusts="product_tax"))


@pytest.mark.parametrize("case", ["transport_regional_investment", "migration_logit",
                                  "taxes_transfers"])
def test_the_exact_jacobian_reaches_the_same_equilibrium(monkeypatch, case):
    build, closure, scenario = CASES[case]
    econ = build()
    exact = cge.solve_period(econ, closure, scenario)
    monkeypatch.setattr(regional_jacobian, "supported", lambda closure: False)
    numeric = cge.solve_period(econ, closure, scenario)
    assert exact.residual < 1e-14
    np.testing.assert_allclose(exact.factor_prices, numeric.factor_prices, rtol=1e-12)
    np.testing.assert_allclose(exact.X, numeric.X, rtol=1e-12)
    assert exact.exchange_rate == pytest.approx(numeric.exchange_rate, rel=1e-12)
