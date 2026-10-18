"""Welfare decomposition: the identity closes, and the terms mean what they say."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import hosoe, toy

_, PLAIN = toy.build()
TT = toy.spec(tariff_account="trf", indirect_tax_account="idt")
CASES = {
    "product_tax": (PLAIN, None, cge.Scenario(product_tax=PLAIN.ptax * 1.5)),
    "endowment": (PLAIN, None, cge.Scenario(factor_supply=PLAIN.fac_supply * [1.05, 1.0])),
    "technology": (PLAIN, None, cge.Scenario(productivity=np.array([1.05, 1.0]))),
    "world_price": (PLAIN, None, cge.Scenario(world_price=np.array([1.2, 1.0]))),
    "margins": (toy.build_with_margins()[1], None, "ptax"),
    "trade_taxes": (toy.build_with_trade_taxes()[1], None, "tariff_and_idt"),
    "cet_curves": (toy.build_with_trade_taxes(model_spec=toy.spec(
        tariff_account="trf", indirect_tax_account="idt", export_rule="cet",
        export_demand=2.0, import_supply=3.0))[1], None,
        cge.Scenario(tariff=np.zeros(2), export_price=np.array([1.1, 1.0]))),
    "wedge": (PLAIN, None, cge.Scenario(consumer_wedge=cge.consumer_wedge(PLAIN, {"c_ind": 0.1}),
                                        government_wedge=cge.consumer_wedge(PLAIN, {"c_agr": 0.05}))),
    "own_account": (toy.build_with_own_production()[1], None,
                    cge.Scenario(activity_scale={"a_own": 1.5})),
    "les_closures": (toy.build_with_institutions(model_spec=toy.spec(household_demand="les"))[1],
                     cge.Closure(investment="fixed_real", government="saving_rate",
                                 foreign_saving_unit="foreign_currency"), "ptax"),
}


def _scenario(econ, sc):
    if sc == "ptax":
        return cge.Scenario(product_tax=econ.ptax * 1.4)
    if sc == "tariff_and_idt":
        return cge.Scenario(tariff=np.zeros(2), output_tax=econ.tauz0 * 2)
    return sc


@pytest.mark.parametrize("name", list(CASES))
def test_the_terms_add_up_to_the_change_in_absorption(name):
    econ, closure, sc = CASES[name]
    w = cge.welfare_decomposition(econ, cge.Scenario(), _scenario(econ, sc), closure=closure,
                                  nodes=6)
    assert abs(w.residual) < 1e-8 * max(abs(w.absorption), 1.0), w
    parts = ["absorption_households", "absorption_own_account", "absorption_government",
             "absorption_investment"]
    assert w[parts].sum() == pytest.approx(w.absorption, abs=1e-12)


def test_without_distortions_an_endowment_is_all_there_is():
    """Hosoe's splcge: no taxes, no trade. The absorption gain is the endowment term."""
    _, econ = hosoe.build_simple()
    w = cge.welfare_decomposition(econ, cge.Scenario(), cge.Scenario(
        factor_supply=econ.fac_supply * np.array([1.1, 1.0])))
    assert w.endowment == pytest.approx(w.absorption, rel=1e-9)
    others = [k for k in w.index if k.startswith("allocative") or k in (
        "technology", "terms_of_trade", "external_account")]
    assert np.allclose(w[others], 0.0, atol=1e-10)
    # Labour is the numeraire, so its wage stays at one: 10% of 25.
    assert w.endowment == pytest.approx(2.5, rel=1e-9)


def test_a_small_country_has_no_terms_of_trade_effect_from_home_shocks():
    econ, _, _ = CASES["cet_curves"]
    small = toy.build_with_trade_taxes(model_spec=toy.spec(
        tariff_account="trf", indirect_tax_account="idt", export_rule="cet"))[1]
    w = cge.welfare_decomposition(small, cge.Scenario(), cge.Scenario(tariff=np.zeros(2)))
    assert w.terms_of_trade == pytest.approx(0.0, abs=1e-10)
    large = cge.welfare_decomposition(econ, cge.Scenario(), cge.Scenario(tariff=np.zeros(2)))
    assert large.terms_of_trade < 0.0          # a large country's imports cost more


def test_a_new_distortion_costs_welfare():
    w = cge.welfare_decomposition(PLAIN, cge.Scenario(), cge.Scenario(
        consumer_wedge=cge.consumer_wedge(PLAIN, {"c_ind": 0.15})))
    assert w.allocative_consumer_wedge < 0.0


def test_hosoe_tariff_abolition():
    """stdcge: an allocative gain and no terms-of-trade effect. lrgcge: the gain is
    there, and the terms of trade take more than it back."""
    abolish = cge.Scenario(tariff=np.zeros(2))
    _, std = hosoe.build()
    w = cge.welfare_decomposition(std, cge.Scenario(), abolish, closure=hosoe.CLOSURE)
    assert w.allocative_tariff > 0 and w.terms_of_trade == pytest.approx(0.0, abs=1e-10)
    assert abs(w.residual) < 1e-8
    _, lrg = hosoe.build(large_country=True)
    w = cge.welfare_decomposition(lrg, cge.Scenario(), abolish, closure=hosoe.CLOSURE)
    assert w.allocative_tariff > 0 and w.terms_of_trade < -w.allocative_tariff
    # The households gain while public consumption, which lost the revenue, falls.
    assert w.absorption_households > 0 > w.absorption_government
