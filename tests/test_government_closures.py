"""Government targets: a deficit target and revenue-neutral tax replacement."""

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import toy

_, INST = toy.build_with_institutions()
CLOSURES = {
    "saving_direct": ck.Closure(government_target="saving", government_adjusts="direct_tax"),
    "saving_product": ck.Closure(government_target="saving", government_adjusts="product_tax"),
    "saving_spending": ck.Closure(government_target="saving", government_adjusts="spending"),
    "revenue_direct": ck.Closure(government_target="revenue", government_adjusts="direct_tax"),
    "revenue_product": ck.Closure(government_target="revenue", government_adjusts="product_tax",
                                  investment="fixed_real", foreign="fixed_rate"),
}


@pytest.mark.parametrize("name", list(CLOSURES))
def test_base_year_and_homogeneity(name):
    rep = ck.base_year_reproduction(INST, CLOSURES[name])
    assert rep["ok"], rep["checks"]
    assert ck.homogeneity_test(INST, CLOSURES[name])["ok"]


@pytest.mark.parametrize("name", list(CLOSURES))
def test_the_target_holds_under_a_shock(name):
    closure = CLOSURES[name]
    sc = (ck.Scenario(factor_supply=INST.fac_supply * [1.05, 1.0])
          if closure.government_adjusts == "product_tax"
          else ck.Scenario(product_tax=INST.ptax * 0.5))
    r = ck.solve_period(INST, closure, sc)
    assert r.converged and abs(r.walras) < 1e-9
    key = "saving" if closure.government_target == "saving" else "revenue"
    base = INST.gov_saving0 if key == "saving" else INST.gov_revenue0
    assert r.government[key] == pytest.approx(base, abs=1e-9)


def test_halving_the_product_tax_raises_direct_tax_rates():
    r = ck.solve_period(INST, CLOSURES["revenue_direct"],
                        ck.Scenario(product_tax=INST.ptax * 0.5))
    assert r.tax_scale > 1.0
    np.testing.assert_allclose(r.hh_direct_tax / r.hh_income, INST.hh_dtax * r.tax_scale,
                               rtol=1e-12)
    assert r.government["product_tax"] < INST.gov_revenue0      # and it fell


def test_a_deficit_target_met_by_spending_cuts_public_consumption():
    plain = ck.solve_period(INST, ck.Closure(), ck.Scenario(product_tax=INST.ptax * 0.5))
    target = ck.solve_period(INST, CLOSURES["saving_spending"],
                             ck.Scenario(product_tax=INST.ptax * 0.5))
    assert target.gov_quantity.sum() < plain.gov_quantity.sum()
    spend = target.pq * target.gov_quantity
    np.testing.assert_allclose(spend / spend.sum(), INST.gov_dem / INST.gov_dem.sum())


def test_an_explicit_target_and_the_numeraire():
    c = ck.Closure(government_target="saving", government_adjusts="direct_tax",
                   government_target_value=12.0)
    r = ck.solve_period(INST, c)
    assert r.government["saving"] == pytest.approx(12.0, abs=1e-9) and r.tax_scale < 1.0  # 12 < 15
    scaled = ck.solve_period(INST, ck.Closure(**{**c.__dict__, "numeraire": 2.0}))
    assert scaled.government["saving"] == pytest.approx(24.0, abs=1e-9)
    assert scaled.tax_scale == pytest.approx(r.tax_scale, rel=1e-10)


def test_misuse_is_refused():
    with pytest.raises(ValueError, match="revenue target needs a tax"):
        ck.Closure(government_target="revenue", government_adjusts="spending")
    with pytest.raises(ValueError, match="second rule"):
        ck.Closure(government="saving_rate", government_target="saving",
                   government_adjusts="spending")
    with pytest.raises(ValueError, match="government_target"):
        ck.Closure(government_target="deficit")
    _, plain = toy.build()                       # no direct tax in the plain toy
    with pytest.raises(ValueError, match="no household pays direct tax"):
        ck.solve_period(plain, CLOSURES["saving_direct"])
