"""Emissions accounts and a carbon price."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy

_, INST = toy.build_with_institutions()
INTER = pd.DataFrame(0.0, index=INST.coms, columns=INST.acts)
INTER.loc["c_ind", "a_agr"], INTER.loc["c_ind", "a_ind"] = 3.0, 12.0
TOTALS = dict(intermediate=INTER, household={"c_ind": 5.0}, government={"c_ind": 1.0},
              process={"a_ind": 2.0})
ECON = cge.add_emissions(INST, **TOTALS)


def test_the_base_year_returns_the_totals():
    e = cge.emissions(cge.solve_period(ECON))
    assert e.to_dict() == pytest.approx({"intermediate": 15.0, "household": 5.0,
                                         "government": 1.0, "process": 2.0, "total": 23.0})


def test_carbon_revenue_is_price_times_taxed_emissions():
    r = cge.solve_period(ECON, scenario=cge.Scenario(carbon_price=0.5))
    e = cge.emissions(r)
    g = r.government
    assert g["carbon_production"] == pytest.approx(0.5 * (e.intermediate + e.process), rel=1e-12)
    assert g["carbon_households"] == pytest.approx(0.5 * e.household, rel=1e-12)
    assert e.total < 23.0                       # a price on carbon cuts it


@pytest.mark.parametrize("closure", [
    cge.Closure(), cge.Closure(foreign="fixed_rate", investment="fixed_real"),
    cge.Closure(government_target="revenue", government_adjusts="product_tax")],
    ids=["default", "fixed", "revenue_neutral"])
def test_identities_hold_with_a_carbon_price(closure):
    sc = cge.Scenario(carbon_price=0.5)
    assert cge.base_year_reproduction(ECON, closure)["ok"]
    assert cge.homogeneity_test(ECON, closure)["ok"]
    r = cge.solve_period(ECON, closure, sc)
    assert abs(r.walras) < 1e-9
    na = cge.national_accounts(r)
    assert abs(na["gap"]) < 1e-9 * na["gdp_expenditure"]


def test_on_production_it_is_an_activity_tax_by_emission_intensity():
    """With Leontief intermediates, taxing the fuel an activity burns is taxing its
    output by the emissions per unit: the two solve to the same equilibrium."""
    no_hh = cge.add_emissions(INST, intermediate=INTER, process={"a_ind": 2.0})
    carbon = cge.solve_period(no_hh, scenario=cge.Scenario(carbon_price=0.5))
    intensity = no_hh.emissions.intensity(no_hh.io)
    import dataclasses
    as_tax = dataclasses.replace(INST, atax=INST.atax + 0.5 * intensity)
    direct = cge.solve_period(as_tax)
    np.testing.assert_allclose(carbon.X, direct.X, rtol=1e-12)
    np.testing.assert_allclose(carbon.pq, direct.pq, rtol=1e-12)


def test_revenue_neutral_recycling_cuts_the_other_tax():
    r = cge.solve_period(ECON, cge.Closure(government_target="revenue",
                                           government_adjusts="product_tax"),
                         cge.Scenario(carbon_price=0.5))
    assert r.tax_scale < 1.0
    assert r.government["revenue"] == pytest.approx(ECON.gov_revenue0, abs=1e-9)


def test_welfare_decomposition_closes_with_carbon_and_productivity():
    w = cge.welfare_decomposition(ECON, cge.Scenario(), cge.Scenario(
        carbon_price=0.5, productivity=np.array([1.03, 1.0]), product_tax=INST.ptax * 0.8))
    assert abs(w.residual) < 1e-8
    w = cge.welfare_decomposition(ECON, cge.Scenario(carbon_price=0.3),
                                  cge.Scenario(carbon_price=0.3, productivity=np.array([1.0, 1.05])))
    assert abs(w.residual) < 1e-8


def test_subtotals_put_the_carbon_price_on_the_path():
    d = cge.subtotals(ECON, cge.Scenario(), cge.Scenario(carbon_price=0.5,
                                                         product_tax=INST.ptax * 0.8),
                      {"carbon": ["carbon_price"], "tax": ["product_tax"]},
                      measure=lambda p: {"em": cge.emissions(p)["total"]}, nodes=4)
    assert d.loc["em", "carbon"] < 0 and abs(d.loc["em", "residual"]) < 1e-8


def test_misuse_is_refused():
    with pytest.raises(ValueError, match="emissions account"):
        cge.solve_period(INST, scenario=cge.Scenario(carbon_price=1.0))
    with pytest.raises(KeyError, match="not in the model"):
        cge.add_emissions(INST, household={"c_oil": 1.0})
    from cgelab.examples import hosoe
    _, simple = hosoe.build_simple()            # government buys nothing in splcge
    with pytest.raises(ValueError, match="records as zero"):
        cge.add_emissions(simple, government={"c_brd": 1.0})


@pytest.mark.parametrize("target,adjusts", [("revenue", "product_tax"), ("revenue", "direct_tax"),
                                            ("saving", "spending")])
def test_welfare_decomposition_under_recycling_closures(target, adjusts):
    """The decomposition must use the rates the solver settled on, not the scenario's."""
    closure = cge.Closure(government_target=target, government_adjusts=adjusts)
    w = cge.welfare_decomposition(ECON, cge.Scenario(), cge.Scenario(carbon_price=0.5),
                                  closure=closure)
    assert abs(w.residual) < 1e-8
    assert w.allocative_carbon_tax < 0
