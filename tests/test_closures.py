"""Government that saves a share of its income, and foreign saving fixed in dollars."""

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import toy

SAVING_RATE = ck.Closure(government="saving_rate")
DOLLARS = ck.Closure(foreign_saving_unit="foreign_currency")
BOTH = ck.Closure(government="saving_rate", foreign_saving_unit="foreign_currency",
                  investment="fixed_real")
BUILDS = [toy.build, toy.build_with_margins, toy.build_with_institutions,
          toy.build_with_trade_taxes]


@pytest.mark.parametrize("closure", [SAVING_RATE, DOLLARS, BOTH], ids=["gov", "usd", "both"])
@pytest.mark.parametrize("build", BUILDS, ids=lambda b: b.__name__)
def test_base_year_and_homogeneity(closure, build):
    _, econ = build()
    rep = ck.base_year_reproduction(econ, closure)
    assert rep["ok"], rep["checks"]
    hom = ck.homogeneity_test(econ, closure)
    assert hom["ok"], hom["checks"]


def _gov_income(econ, res):
    return float(res.savings - res.foreign_saving
                 - ((res.hh_income - res.hh_direct_tax) * econ.hh_savrate).sum()
                 + res.pq @ res.gov_quantity)


def test_government_spends_its_income_less_a_fixed_saving_rate():
    _, econ = toy.build()
    res = ck.solve_period(econ, SAVING_RATE, ck.Scenario(product_tax=econ.ptax * 2))
    assert res.converged and abs(res.walras) < 1e-9
    income = _gov_income(econ, res)
    assert float(res.pq @ res.gov_quantity) == pytest.approx(
        (1 - econ.gov_savrate) * income, rel=1e-10)
    # in fixed value shares
    spend = res.pq * res.gov_quantity
    np.testing.assert_allclose(spend / spend.sum(), econ.gov_dem / econ.gov_dem.sum())


def test_a_tax_rise_raises_public_consumption_only_under_the_saving_rate():
    _, econ = toy.build()
    sc = ck.Scenario(product_tax=econ.ptax * 2)
    fixed = ck.solve_period(econ, ck.Closure(), sc)
    share = ck.solve_period(econ, SAVING_RATE, sc)
    np.testing.assert_allclose(fixed.gov_quantity, econ.gov_dem / (1 + econ.ptax))
    assert share.gov_quantity.sum() > fixed.gov_quantity.sum()


def test_foreign_saving_in_dollars_moves_with_the_exchange_rate():
    _, econ = toy.build()
    sc = ck.Scenario(world_price=np.array([1.3, 1.0]))
    res = ck.solve_period(econ, DOLLARS, sc)
    assert res.converged and abs(res.walras) < 1e-9
    assert res.foreign_saving == pytest.approx(res.exchange_rate * econ.foreign_sav, rel=1e-12)
    numeraire = ck.solve_period(econ, ck.Closure(), sc)
    assert numeraire.foreign_saving == pytest.approx(econ.foreign_sav, rel=1e-12)
    assert res.exchange_rate != pytest.approx(numeraire.exchange_rate, rel=1e-6)


def test_unknown_closure_values_are_refused():
    with pytest.raises(ValueError, match="foreign_saving_unit"):
        ck.Closure(foreign_saving_unit="dollars")
    with pytest.raises(ValueError, match="government"):
        ck.Closure(government="balanced")


def test_an_undetermined_exchange_rate_is_refused():
    """Fixed export shares, a Cobb-Douglas Armington and foreign saving in the
    numeraire leave the exchange rate free: 0.5 and 2 were both exact equilibria,
    with the same output and factor prices."""
    sam, _ = toy.build()
    econ = ck.calibrate(sam, toy.spec(sigma_armington=1.0))
    with pytest.raises(ValueError, match="not determined"):
        ck.solve_period(econ)
    # Each of these restores a channel, and the solve goes through.
    assert ck.solve_period(econ, DOLLARS).converged
    assert ck.solve_period(econ, ck.Closure(foreign="fixed_rate")).converged
    mixed = ck.calibrate(sam, toy.spec(sigma_armington={"c_agr": 1.0, "c_ind": 2.0}))
    assert ck.solve_period(mixed).converged
    cet = ck.calibrate(sam, toy.spec(sigma_armington=1.0, export_rule="cet"))
    assert ck.solve_period(cet).converged
