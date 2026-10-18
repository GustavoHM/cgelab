"""National accounts, price indices and the government account."""

import json
from pathlib import Path

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import hosoe, toy

_, PLAIN = toy.build()
CASES = {
    "plain": (PLAIN, cge.Closure(), cge.Scenario(product_tax=PLAIN.ptax * 1.3)),
    "margins": (toy.build_with_margins()[1], cge.Closure(),
                cge.Scenario(world_price=np.array([1.2, 1.0, 1.0]))),
    "institutions": (toy.build_with_institutions()[1],
                     cge.Closure(investment="fixed_real", foreign="fixed_rate"),
                     cge.Scenario(fdi={"f_cap": 3.0})),
    "trade_taxes": (toy.build_with_trade_taxes()[1], cge.Closure(government="saving_rate"),
                    cge.Scenario(tariff=np.zeros(2))),
    "own_account": (toy.build_with_own_production()[1], cge.Closure(),
                    cge.Scenario(activity_scale={"a_own": 1.5})),
    "wedges": (PLAIN, cge.Closure(), cge.Scenario(
        consumer_wedge=cge.consumer_wedge(PLAIN, {"c_ind": 0.1}),
        government_wedge=cge.consumer_wedge(PLAIN, {"c_agr": -0.05}))),
    "cet_curves": (toy.build(model_spec=toy.spec(export_rule="cet", export_demand=2.0,
                                                 import_supply=2.0))[1],
                   cge.Closure(), cge.Scenario(export_price=np.array([1.2, 0.9]))),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gdp_from_both_sides_agrees_in_every_equilibrium(name):
    econ, closure, scenario = CASES[name]
    for sc in (None, scenario):
        na = cge.national_accounts(cge.solve_period(econ, closure, sc))
        assert abs(na["gap"]) < 1e-9 * na["gdp_expenditure"], na


def test_base_year_gdp_is_the_sams():
    na = cge.national_accounts(cge.solve_period(PLAIN))
    # households 135, government 20, investment 50, exports 30, imports 40
    assert na["gdp_expenditure"] == pytest.approx(195.0, rel=1e-12)
    assert na["factor_income"] == pytest.approx(110.0 + 60.0, rel=1e-12)  # labour, capital
    assert na["taxes_on_products"] + na["taxes_on_production"] == pytest.approx(25.0)


@pytest.mark.parametrize("name", list(CASES))
def test_index_number_identities(name):
    econ, closure, scenario = CASES[name]
    period = cge.solve_period(econ, closure, scenario)
    pi = cge.price_indices(period).drop("terms_of_trade")
    np.testing.assert_allclose(pi.fisher * pi.quantity_fisher, pi.value, rtol=1e-12)
    np.testing.assert_allclose(pi.laspeyres * pi.quantity_paasche, pi.value, rtol=1e-12)
    np.testing.assert_allclose(pi.paasche * pi.quantity_laspeyres, pi.value, rtol=1e-12)
    # Nominal GDP over the Paasche deflator is GDP at base-year prices.
    na = cge.national_accounts(period)
    assert na["gdp_expenditure"] / pi.loc["gdp", "paasche"] == pytest.approx(
        cge.real_gdp(period)["expenditure"], rel=1e-12)


def test_indices_at_the_base_year_and_against_oneself_are_one():
    base = cge.solve_period(PLAIN)
    np.testing.assert_allclose(cge.price_indices(base).fillna(1.0), 1.0, atol=1e-12)
    shock = cge.solve_period(PLAIN, scenario=CASES["plain"][2])
    np.testing.assert_allclose(cge.price_indices(shock, reference=shock).fillna(1.0), 1.0,
                               atol=1e-12)


def test_a_dearer_export_improves_the_terms_of_trade():
    econ, closure, sc = CASES["cet_curves"]
    tot = cge.price_indices(cge.solve_period(econ, closure, sc)).loc["terms_of_trade"]
    assert tot.fisher > 1.0


def test_government_account_closes_on_the_saving_that_financed_investment():
    for econ, closure, sc in CASES.values():
        r = cge.solve_period(econ, closure, sc)
        g = cge.government_account(r)
        assert g.revenue == pytest.approx(g[["product_tax", "activity_tax", "tariff",
                                             "output_tax", "direct_tax", "factor_income",
                                             "wedge"]].sum())
        assert g.saving == pytest.approx(
            r.savings - r.household_saving - r.forced_saving - r.foreign_saving, abs=1e-9)


REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "hosoe_tariff_abolition.json").read_text())


@pytest.mark.parametrize("model,large", [("stdcge", False), ("lrgcge", True)])
def test_gdp_and_terms_of_trade_against_the_hosoe_transcription(model, large):
    """Built from the transcription's own variables, not from cgelab's."""
    goods = ("BRD", "MLK")
    _, econ = hosoe.build(large_country=large)
    r = cge.solve_period(econ, hosoe.CLOSURE, cge.Scenario(tariff=np.zeros(2)))
    h, h0 = REFERENCE[model]["abolicao"], REFERENCE[model]["base"]
    gdp = sum(h["pq"][g] * (h["Xp"][g] + h["Xg"][g] + h["Xv"][g]) + h["pe"][g] * h["E"][g]
              - h["pm"][g] * h["M"][g] for g in goods)
    assert cge.national_accounts(r)["gdp_expenditure"] == pytest.approx(gdp, rel=1e-9)
    tot = (sum(h["pWe"][g] * h0["E"][g] for g in goods) / sum(h0["E"][g] for g in goods)
           / (sum(h["pWm"][g] * h0["M"][g] for g in goods) / sum(h0["M"][g] for g in goods)))
    assert cge.price_indices(r).loc["terms_of_trade", "laspeyres"] == pytest.approx(tot, rel=1e-9)
    if large:
        assert tot < 1.0          # a large country pays for its export drive abroad
