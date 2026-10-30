"""Route margins and the global transport pool inside the regional equilibrium."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def transport_economy():
    return regions.build_with_transport()


def test_transport_base_year_reproduces_sam_and_gtap_identities():
    econ = transport_economy()[1]
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-8
    np.testing.assert_allclose(result.X, [100.0, 12.0, 100.0, 12.0], atol=1e-7)
    np.testing.assert_allclose(result.Q, [100.0, 12.0, 100.0, 12.0], atol=1e-7)
    np.testing.assert_allclose(result.bilateral[:, :, 0], [[45.0, 18.0], [18.0, 45.0]],
                               atol=1e-7)
    purchaser = (result.bilateral[:, :, 0] + result.bilateral_imports[:, :, 0]
                 + result.transport["vtwr"][:, :2, 0])
    np.testing.assert_allclose(purchaser, [[62.0, 20.0], [20.0, 62.0]], atol=1e-7)
    transport = result.transport
    np.testing.assert_allclose(transport["viws"], transport["vxwd"] + transport["vtwr"])
    assert transport["qt"] == pytest.approx(24.0, abs=1e-7)
    assert transport["pt"] == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(transport["qst"], [[0.0, 12.0], [0.0, 12.0]], atol=1e-7)
    assert transport["vtwr"].sum() == pytest.approx(transport["qst"].sum(), abs=1e-7)


def test_transport_shock_preserves_walras_and_margin_value():
    econ = transport_economy()[1]
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("r1:a")] = 1.2
    result = cge.solve_period(econ, scenario=cge.Scenario(productivity=tfp))
    assert result.converged and abs(result.walras) < 1e-8
    transport = result.transport
    assert transport["qt"] > 0
    assert transport["vtwr"].sum() == pytest.approx(transport["pt"] * transport["qt"])
    service_prices = result.pd[econ.regional.origin_index]
    assert (service_prices * transport["qst"]).sum() == pytest.approx(transport["vtwr"].sum())


def test_transport_is_homogeneous_of_degree_zero():
    econ = transport_economy()[1]
    base = cge.solve_period(econ)
    scaled = cge.solve_period(econ, cge.Closure(numeraire=2.0))
    np.testing.assert_allclose(scaled.X, base.X, rtol=1e-7, atol=1e-7)
    assert scaled.transport["qt"] == pytest.approx(base.transport["qt"], rel=1e-7)
    assert scaled.transport["pt"] == pytest.approx(2.0 * base.transport["pt"], rel=1e-7)
    assert abs(scaled.walras) < 1e-8


def test_transport_reconciliation_errors_are_named():
    sam, econ = transport_economy()
    margins = econ.regional.transport
    bad = cge.TransportMargins(margins.regions, margins.commodities, margins.destinations,
                               margins.fob * 1.01, margins.transport, margins.supply)
    with pytest.raises(ValueError, match="domestic sales|imports"):
        cge.calibrate(sam, econ.spec, transport=bad)
    with pytest.raises(ValueError, match="require ModelSpec.regions"):
        cge.calibrate(toy.build()[0], toy.spec(), transport=margins)


@pytest.mark.parametrize("closure", [cge.Closure(), cge.Closure(investment="fixed_real")])
def test_regional_investment_at_different_base_prices_reproduces_the_sam(closure):
    sam, econ = regions.build_with_transport(regional_investment=True)
    assert not np.allclose(econ.regional.base_price[1], econ.regional.base_price[2])
    result = cge.solve_period(econ, closure)
    assert result.converged and abs(result.walras) < 1e-8
    np.testing.assert_allclose(result.X, econ.X0, atol=1e-7)
    spent = result.real["destination_price"] * result.real["inv_destination_quantity"]
    np.testing.assert_allclose(spent[:, 0], [0.0, 20.0, 20.0], atol=1e-7)
    assert sam.block(econ.coms, ["r2:s_i"]).sum() == pytest.approx(spent[1].sum())


@pytest.mark.parametrize("rule", ["fixed_shares", "cet"])
def test_an_exported_transport_service_reproduces_the_sam_and_closes(rule):
    _, econ = regions.build_with_transport(service_exports=True, export_rule=rule,
                                           sigma_cet=3.0)
    base = cge.solve_period(econ)
    assert abs(base.walras) < 1e-8
    np.testing.assert_allclose(base.Q, [100.0, 14.0, 100.0, 12.0], atol=1e-7)
    np.testing.assert_allclose(base.exports, [10.0, 2.0, 10.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(base.transport["qst"], [[0.0, 12.0], [0.0, 12.0]], atol=1e-7)
    tfp = np.ones(econ.n_act)
    tfp[1] = 1.1
    shocked = cge.solve_period(econ, scenario=cge.Scenario(
        productivity=tfp, world_price=np.array([1.2, 1.0]),
        export_price=np.array([1.0, 1.3]) if rule == "cet" else None))
    assert abs(shocked.walras) < 1e-8
    assert abs(cge.national_accounts(shocked)["gap"]) < 1e-8
