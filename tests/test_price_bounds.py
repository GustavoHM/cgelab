"""Price bounds: a ceiling held by a subsidy, a floor by purchases, a wage floor by unemployment."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

SCARCE = cge.Scenario(productivity=np.array([0.8, 1.0]))    # dearer agriculture
ABUNDANT = cge.Scenario(productivity=np.array([1.3, 1.0]))  # cheaper agriculture


def real_price(result, commodity):
    """Purchaser price over the purchaser index, in base-year units: what a bound reads."""
    from cgelab.solve import _laspeyres_purchaser
    econ = result.econ
    i = econ.commodity(commodity)
    return result.pq[i] / (_laspeyres_purchaser(econ, result.pq) * (1.0 + econ.ptax[i]))


def test_slack_bounds_change_nothing():
    _, econ = toy.build()
    free = cge.solve_period(econ, scenario=SCARCE)
    bounded = cge.solve_period(econ, scenario=SCARCE.with_(price_ceiling={"c_agr": 2.0},
                                                           price_floor={"c_ind": 0.5}))
    np.testing.assert_allclose(bounded.X, free.X, rtol=1e-12)
    assert bounded.price_bounds["ceiling"]["c_agr"]["subsidy"] == 0.0
    assert bounded.price_bounds["floor"]["c_ind"]["purchases"] == 0.0
    assert bounded.stock_quantity is not None and not bounded.stock_quantity.any()


def test_a_binding_ceiling_is_held_by_a_subsidy():
    _, econ = toy.build()
    free = cge.solve_period(econ, scenario=SCARCE)
    assert real_price(free, "c_agr") > 1.1
    result = cge.solve_period(econ, scenario=SCARCE.with_(price_ceiling={"c_agr": 1.1}))
    assert result.converged and abs(result.walras) < 1e-10
    cap = result.price_bounds["ceiling"]["c_agr"]
    assert real_price(result, "c_agr") == pytest.approx(1.1, rel=1e-12)
    assert cap["price"] == pytest.approx(cap["bound"], rel=1e-12)
    assert cap["subsidy"] > 0.01
    i = econ.commodity("c_agr")
    assert cap["cost"] == pytest.approx(cap["subsidy"] * result.pbase[i] * result.absorption[i])
    gov = cge.government_account(result)
    assert gov["price_subsidy"] == pytest.approx(cap["cost"])
    accounts = cge.national_accounts(result)
    assert abs(accounts["gap"]) < 1e-8
    assert accounts["price_subsidy"] == pytest.approx(cap["cost"])


def test_a_binding_ceiling_is_the_tax_cut_it_amounts_to():
    """The known solution: the solved subsidy, entered as a product-tax cut, is the same economy."""
    _, econ = toy.build()
    result = cge.solve_period(econ, scenario=SCARCE.with_(price_ceiling={"c_agr": 1.1}))
    rates = econ.ptax.copy()
    rates[econ.commodity("c_agr")] -= result.price_bounds["ceiling"]["c_agr"]["subsidy"]
    cut = cge.solve_period(econ, scenario=SCARCE.with_(product_tax=rates, product_tax_firm=rates))
    np.testing.assert_allclose(cut.X, result.X, rtol=1e-12)
    np.testing.assert_allclose(cut.pq, result.pq, rtol=1e-12)
    np.testing.assert_allclose(cut.hh_income, result.hh_income, rtol=1e-12)
    assert cut.government["saving"] == pytest.approx(result.government["saving"], abs=1e-10)


def test_a_binding_floor_is_held_by_government_purchases():
    _, econ = toy.build()
    free = cge.solve_period(econ, scenario=ABUNDANT)
    assert real_price(free, "c_agr") < 0.871
    result = cge.solve_period(econ, scenario=ABUNDANT.with_(price_floor={"c_agr": 0.871}))
    assert result.converged and abs(result.walras) < 1e-10
    floor = result.price_bounds["floor"]["c_agr"]
    i = econ.commodity("c_agr")
    assert real_price(result, "c_agr") == pytest.approx(0.871, rel=1e-12)
    assert floor["purchases"] > 1.0
    assert result.stock_quantity[i] == pytest.approx(floor["purchases"])
    assert floor["cost"] == pytest.approx(result.pq[i] * floor["purchases"])
    gov = cge.government_account(result)
    assert gov["price_support"] == pytest.approx(floor["cost"])
    assert gov["saving"] == pytest.approx(gov["revenue"] - gov["transfers"] - gov["consumption"]
                                          - gov["absorbed"] - gov["price_support"])
    accounts = cge.national_accounts(result)
    assert abs(accounts["gap"]) < 1e-8
    assert accounts["government"] == pytest.approx(gov["consumption"] + floor["cost"])


@pytest.mark.parametrize("scenario", [
    SCARCE.with_(price_ceiling={"c_agr": 1.1}),
    ABUNDANT.with_(price_floor={"c_agr": 0.871}),
], ids=["ceiling", "floor"])
def test_price_bounds_are_homogeneous_of_degree_zero(scenario):
    _, econ = toy.build()
    one = cge.solve_period(econ, scenario=scenario)
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0), scenario)
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10)
    for kind, bounds in one.price_bounds.items():
        for com, b in bounds.items():
            other = two.price_bounds[kind][com]
            assert other["bound"] == pytest.approx(2.0 * b["bound"])
            assert other["cost"] == pytest.approx(2.0 * b["cost"])


def labour_economy():
    """Capital is the numeraire, so the labour market can be an equation."""
    _, econ = toy.build(toy.spec(numeraire_factor="f_cap"))
    supply = econ.fac_supply.copy()
    supply[econ.factor("f_cap")] *= 0.8                # less capital: a lower market wage
    return econ, cge.Scenario(factor_supply=supply)


def test_the_base_year_sits_on_the_wage_floor():
    econ, _ = labour_economy()
    result = cge.solve_period(econ, cge.Closure(labour="wage_floor"))
    assert result.attempt == "cold" and result.residual < 1e-12
    assert result.unemployment == pytest.approx(0.0, abs=1e-9)


def test_a_slack_wage_floor_is_full_employment():
    econ, scenario = labour_economy()
    full = cge.solve_period(econ, scenario=scenario)
    assert full.real_wage < 0.95
    floor = cge.solve_period(econ, cge.Closure(labour="wage_floor", real_wage=0.85), scenario)
    np.testing.assert_allclose(floor.X, full.X, rtol=1e-10)
    assert floor.real_wage == pytest.approx(full.real_wage, rel=1e-10)
    assert floor.unemployment == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("level", [1.0, 0.95])
def test_a_binding_wage_floor_is_the_fixed_real_wage(level):
    econ, scenario = labour_economy()
    floor = cge.solve_period(econ, cge.Closure(labour="wage_floor", real_wage=level), scenario)
    fixed = cge.solve_period(econ, cge.Closure(labour="fixed_real_wage", real_wage=level), scenario)
    assert floor.converged and abs(floor.walras) < 1e-10
    np.testing.assert_allclose(floor.X, fixed.X, rtol=1e-10)
    assert floor.real_wage == pytest.approx(level, rel=1e-12)
    assert floor.unemployment == pytest.approx(fixed.unemployment, rel=1e-9)
    assert floor.unemployment > 1.0


def test_the_wage_floor_is_homogeneous_of_degree_zero():
    econ, scenario = labour_economy()
    one = cge.solve_period(econ, cge.Closure(labour="wage_floor"), scenario)
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0, labour="wage_floor"), scenario)
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10)
    assert two.unemployment == pytest.approx(one.unemployment, rel=1e-10)


def test_price_bounds_are_refused_where_they_cannot_hold():
    _, econ = toy.build()
    with pytest.raises(KeyError, match="c_none"):
        cge.solve_period(econ, scenario=cge.Scenario(price_ceiling={"c_none": 1.0}))
    with pytest.raises(ValueError, match="positive"):
        cge.solve_period(econ, scenario=cge.Scenario(price_floor={"c_agr": 0.0}))
    with pytest.raises(ValueError, match="not below its ceiling"):
        cge.solve_period(econ, scenario=cge.Scenario(price_floor={"c_agr": 1.0},
                                                     price_ceiling={"c_agr": 1.0}))
    with pytest.raises(ValueError, match="price bounds"):
        cge.to_jump(econ, scenario=cge.Scenario(price_ceiling={"c_agr": 1.1}))
    with pytest.raises(ValueError, match="numeraire"):
        cge.solve_period(econ, cge.Closure(labour="wage_floor"))   # f_lab is the numeraire
    labour, _ = labour_economy()
    with pytest.raises(ValueError, match="wage_floor"):
        cge.to_jump(labour, cge.Closure(labour="wage_floor"))
    _, regional = regions.build()
    with pytest.raises(ValueError, match="price_floor"):
        cge.solve_period(regional, scenario=cge.Scenario(price_floor={"r1:c": 1.0}))
