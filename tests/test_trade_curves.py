"""Large-country trade: a demand curve for exports and a supply curve for imports.

The form is Hosoe's ``lrgcge``: ``E / E0 = (pWe / shift)^(-η)`` and
``M / M0 = (pWm / shift)^ψ``, with the scenario's ``export_price`` and
``world_price`` as the shifts.
"""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

SHOCK = dict(export_price=np.array([1.15, 0.9]))
CLOSURES = [
    cge.Closure(),
    cge.Closure(foreign="fixed_rate"),
    cge.Closure(investment="fixed_real"),
    cge.Closure(investment="fixed_real", foreign="fixed_rate"),
]


def large(eta=2.0, psi=2.0, build=toy.build):
    return build(model_spec=toy.spec(export_rule="cet", export_demand=eta,
                                     import_supply=psi))[1]


@pytest.mark.parametrize("build", [toy.build, toy.build_with_margins,
                                   toy.build_with_own_production])
def test_the_base_year_reproduces_on_the_curves(build):
    econ = large(build=build)
    rep = cge.base_year_reproduction(econ)
    assert rep["ok"], rep
    base = cge.solve_period(econ)
    np.testing.assert_allclose(base.world_export_price, 1.0, atol=1e-10)
    np.testing.assert_allclose(base.world_import_price, 1.0, atol=1e-10)


def test_homogeneous_of_degree_zero():
    rep = cge.homogeneity_test(large(), factor=3.0)
    assert rep["ok"], rep


@pytest.mark.parametrize("closure", CLOSURES, ids=lambda c: c.describe()[12:])
def test_walras_holds_under_a_shock(closure):
    econ = large()
    res = cge.solve_period(econ, closure,
                           cge.Scenario(product_tax=econ.ptax * 1.2, **SHOCK))
    assert res.converged
    assert abs(res.walras) < 1e-9


def test_volumes_sit_on_the_curves():
    eta, psi = 1.5, 3.0
    econ = large(eta, psi)
    world = np.array([1.1, 0.95])
    res = cge.solve_period(econ, scenario=cge.Scenario(world_price=world, **SHOCK))
    np.testing.assert_allclose(
        res.exports / econ.exp0,
        (res.world_export_price / SHOCK["export_price"]) ** (-eta), rtol=1e-9)
    np.testing.assert_allclose(
        res.imports / econ.imp0, (res.world_import_price / world) ** psi, rtol=1e-9)


def test_a_demand_shift_is_shared_between_price_and_volume():
    """A rise in foreign demand raises the world price by less than the shift, and
    a flatter curve passes less of it into the price."""
    pe = np.array([1.0, 1.2])
    taker = cge.solve_period(large(None, None), scenario=cge.Scenario(export_price=pe))
    prices = []
    for eta in (0.5, 4.0):
        res = cge.solve_period(large(eta, None), scenario=cge.Scenario(export_price=pe))
        prices.append(res.world_export_price[1])
        assert 1.0 < res.world_export_price[1] < 1.2
        assert res.exports[1] < taker.exports[1]
    assert prices[0] < prices[1]


def test_a_steep_curve_tends_to_the_price_taker():
    sc = cge.Scenario(world_price=np.array([1.1, 1.0]), **SHOCK)
    taker = cge.solve_period(large(None, None), scenario=sc)
    steep = cge.solve_period(large(1e7, 1e7), scenario=sc)
    np.testing.assert_allclose(steep.exports, taker.exports, rtol=1e-5)
    np.testing.assert_allclose(steep.imports, taker.imports, rtol=1e-5)
    np.testing.assert_allclose(steep.factor_prices, taker.factor_prices, rtol=1e-5)


def test_a_mapping_leaves_the_other_commodities_price_takers():
    econ = toy.build(model_spec=toy.spec(export_rule="cet",
                                         export_demand={"c_ind": 1.0}))[1]
    res = cge.solve_period(econ, scenario=cge.Scenario(**SHOCK))
    j = econ.commodity("c_agr")
    assert res.world_export_price[j] == pytest.approx(SHOCK["export_price"][j], abs=1e-14)
    assert res.world_export_price[econ.commodity("c_ind")] != pytest.approx(0.9, abs=1e-4)


def test_a_path_on_the_curves_keeps_walras():
    path = cge.run_path(large(), years=range(2020, 2031),
                        scenario=lambda y, t: cge.Scenario(
                           export_price=np.array([1.0 + 0.01 * t, 1.0])),
                        growth=cge.Growth(labour=0.01, tfp=0.01))
    assert path.converged
    assert path.worst_walras < 1e-9


@pytest.mark.parametrize("value", [0.0, -1.0, {"c_ind": 0.0}])
def test_a_non_positive_elasticity_is_refused(value):
    with pytest.raises(ValueError, match="positive elasticity"):
        toy.spec(export_rule="cet", export_demand=value)
    with pytest.raises(ValueError, match="positive elasticity"):
        toy.spec(import_supply=value)


def test_export_demand_needs_cet():
    with pytest.raises(ValueError, match="export_rule='cet'"):
        toy.spec(export_demand=2.0)


def test_import_supply_works_under_fixed_export_shares():
    econ = toy.build(model_spec=toy.spec(import_supply=2.0))[1]
    res = cge.solve_period(econ, scenario=cge.Scenario(world_price=np.array([1.2, 1.0])))
    assert res.converged and abs(res.walras) < 1e-9
    assert 1.0 < res.world_import_price[0] < 1.2


def test_the_regional_slice_refuses_the_curves():
    with pytest.raises(ValueError, match="price taker abroad"):
        regions.build(model_spec=regions.spec(import_supply=2.0))


def test_to_jump_writes_the_curves(tmp_path):
    econ = toy.build(model_spec=toy.spec(export_rule="cet",
                                         export_demand={"c_ind": 0.8},
                                         import_supply=3.0))[1]
    dados = cge.to_jump(econ, path=tmp_path / "m.jl")
    assert dados["export_demand"] == pytest.approx([0.0, 0.8])
    assert dados["import_supply"] == pytest.approx([3.0, 3.0])
    assert dados["exp0"] == pytest.approx(list(econ.exp0))
    assert "pwe" in (tmp_path / "m.jl").read_text()
