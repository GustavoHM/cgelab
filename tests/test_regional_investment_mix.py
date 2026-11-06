"""Scenario.investment_mix in the regional solver: investment buys by origin.

The mix names regional commodities (``core:c_ind``). Its share of the budget
buys that region's good directly, so no origin CES or Armington nest splits
it again. Household, government and intermediate demand keep the destination
composite.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

REGIONS = ("core", "rob")
SHARED = {"gov", "s_i", "row"}
TRADE = [
    # Each region's first household buys 10 of the other region's industry
    # in place of 10 of its own, and each industry buys 10 of the other
    # region's agriculture in place of its own. Row and column sums are kept.
    ("core:c_ind", "core:hh_1", -10.0), ("rob:c_ind", "core:hh_1", 10.0),
    ("rob:c_ind", "rob:hh_1", -10.0), ("core:c_ind", "rob:hh_1", 10.0),
    ("core:c_agr", "core:a_ind", -10.0), ("rob:c_agr", "core:a_ind", 10.0),
    ("rob:c_agr", "rob:a_ind", -10.0), ("core:c_agr", "rob:a_ind", 10.0),
]


def two_region_toy():
    """The toy economy in two regions, ``rob`` twice the size of ``core``."""
    def name(region, account):
        return account if account in SHARED else f"{region}:{account}"

    rows = [(name(region, r), name(region, c), v * scale)
            for region, scale in (("core", 1.0), ("rob", 2.0))
            for r, c, v in toy._BASE] + TRADE
    sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    sam.check_balance(abs_tol=1e-9)
    spec = cge.ModelSpec(
        regions=REGIONS,
        activities=[f"{r}:a_{m}" for r in REGIONS for m in ("agr", "ind")],
        commodities=[f"{r}:c_{m}" for r in REGIONS for m in ("agr", "ind")],
        households=[f"{r}:hh_{h}" for r in REGIONS for h in (1, 2)],
        factors=tuple(f"{r}:f_{f}" for r in REGIONS for f in ("lab", "cap")),
        numeraire_factor="rob:f_lab", government="gov", investment="s_i",
        rest_of_world="row", sigma_va=0.8, sigma_armington=2.0,
        sigma_interregional=3.0, margins=False)
    return cge.calibrate(sam, spec)


def mix_of(econ, weights):
    mix = np.zeros(econ.n_com)
    for com, w in weights.items():
        mix[econ.commodity(com)] = w
    return mix


CLOSURES = [
    cge.Closure(),
    cge.Closure(investment="fixed_real"),
    cge.Closure(government="nominal_fixed"),
    cge.Closure(government="saving_rate"),
    cge.Closure(investment="fixed_real", government="saving_rate"),
    cge.Closure(government_target="saving", government_adjusts="product_tax"),
]


def test_one_origin_raises_its_own_activity_and_not_the_other():
    econ = two_region_toy()
    closure = cge.Closure(investment="fixed_real")
    level = 1.2 * econ.inv_dem.sum()
    base = cge.solve_period(econ, closure, cge.Scenario(investment_level=level))
    shock = cge.solve_period(econ, closure, cge.Scenario(
        investment_level=level, investment_mix=mix_of(econ, {"core:c_ind": 1.0})))
    assert shock.converged and abs(shock.walras) < 1e-8
    core, rob = econ.activity("core:a_ind"), econ.activity("rob:a_ind")
    assert shock.X[core] > base.X[core]
    assert shock.X[rob] < base.X[rob]
    # Nothing of the budget reaches another origin or the composite.
    bought = shock.investment_quantity
    assert bought[econ.commodity("core:c_ind")] > 0
    others = [i for i in range(econ.n_com) if i != econ.commodity("core:c_ind")]
    np.testing.assert_allclose(bought[others], 0.0, atol=1e-12)
    np.testing.assert_allclose(shock.real["inv_destination_quantity"], 0.0, atol=1e-12)


def test_fixed_real_shares_are_held_exactly():
    econ = two_region_toy()
    reg = econ.regional
    level = 150.0
    weights = {"core:c_ind": 0.75, "rob:c_ind": 0.20, "core:c_agr": 0.05}
    result = cge.solve_period(econ, cge.Closure(investment="fixed_real"), cge.Scenario(
        investment_level=level, investment_mix=mix_of(econ, weights)))
    assert result.converged and abs(result.walras) < 1e-8
    for com, w in weights.items():
        i = econ.commodity(com)
        c = reg.local_commodities.index(com.split(":")[1])
        # Valued at base-year purchaser prices: the origin price is one.
        assert result.investment_quantity[i] * (1.0 + reg.ptax[c]) == pytest.approx(w * level)
    assert result.real["inv_real"] == pytest.approx(level, rel=1e-10)
    origin = result.real["inv_origin_quantity"]
    assert origin.sum() == pytest.approx(result.investment_quantity.sum(), rel=1e-12)


def test_savings_driven_value_shares_follow_the_mix():
    econ = two_region_toy()
    reg = econ.regional
    weights = {"core:c_ind": 0.75, "rob:c_ind": 0.25}
    result = cge.solve_period(econ, scenario=cge.Scenario(
        investment_mix=mix_of(econ, weights), productivity=np.full(econ.n_act, 1.05)))
    assert result.converged and abs(result.walras) < 1e-8
    value = (result.p_domestic * (1.0 + reg.ptax)[None, :]
             * result.real["inv_origin_quantity"])
    assert value.sum() == pytest.approx(result.savings, rel=1e-10)
    for com, w in weights.items():
        r = REGIONS.index(com.split(":")[0])
        c = reg.local_commodities.index(com.split(":")[1])
        assert value[r, c] / value.sum() == pytest.approx(w, rel=1e-10)


@pytest.mark.parametrize("closure", CLOSURES, ids=lambda c: c.describe())
def test_a_mix_closes_every_account(closure):
    econ = two_region_toy()
    mix = mix_of(econ, {"core:c_ind": 0.5, "rob:c_agr": 0.3, "rob:c_ind": 0.2})
    result = cge.solve_period(econ, closure, cge.Scenario(investment_mix=mix))
    assert result.converged
    assert abs(result.walras) < 1e-8
    # Domestic sales plus exports is output, origin by origin.
    np.testing.assert_allclose(result.domestic_sales + result.exports, result.Q, rtol=1e-9)


def test_without_a_mix_nothing_is_bought_by_origin():
    econ = two_region_toy()
    result = cge.solve_period(econ)
    np.testing.assert_allclose(result.real["inv_origin_quantity"], 0.0)
    np.testing.assert_allclose(result.X, econ.X0, rtol=1e-9)


def test_invalid_mixes_are_refused():
    econ = two_region_toy()
    with pytest.raises(ValueError, match="investment_mix"):
        cge.solve_period(econ, scenario=cge.Scenario(investment_mix=np.array([0.5, 0.5])))
    _, transport = regions.build_with_transport()
    mix = np.zeros(transport.n_com)
    mix[0] = 1.0
    with pytest.raises(ValueError, match="route margins"):
        cge.solve_period(transport, scenario=cge.Scenario(investment_mix=mix))
