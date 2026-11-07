"""An export tax is paid out of the border price, and it is neither the tariff nor a product tax."""

import dataclasses

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy

SAM, ECON = toy.build_with_export_tax()
FIXED = toy.spec(tariff_account="trf", export_tax_account="etx")


def test_the_rates_are_revenue_over_exports_at_the_border():
    np.testing.assert_allclose(ECON.te0, [0.5 / 10.0, 1.0 / 20.0])
    np.testing.assert_allclose(ECON.tm0, [1.0 / 15.0, 2.0 / 25.0])
    assert not any("etx" in (a, b) for a, b, _ in ECON.diagnostics["unmodelled_flows"])


@pytest.mark.parametrize("rule", ["cet", "fixed_shares"])
def test_a_calibrated_export_tax_reproduces_the_base_year(rule):
    econ = ECON if rule == "cet" else toy.build_with_export_tax(FIXED)[1]
    period = cge.solve_period(econ)
    assert period.converged and abs(period.walras) < 1e-12
    # Exports are in border units: the rest of the world still pays 10 and 20.
    np.testing.assert_allclose(period.exports, [10.0, 20.0], atol=1e-10)
    np.testing.assert_allclose(period.pe, 1.0, atol=1e-12)
    np.testing.assert_allclose(period.Q, [100.0, 200.0], atol=1e-10)
    assert period.government["export_tax"] == pytest.approx(1.5, abs=1e-12)
    # What buyers pay at home is the plain economy's.
    _, plain = toy.build()
    np.testing.assert_allclose(econ.A0 * (1 + econ.ptax), plain.A0 * (1 + plain.ptax))


def test_reference_rates_are_the_named_account():
    named = cge.solve_period(ECON, scenario=cge.Scenario(export_tax=np.zeros(2)))
    sam = toy._frame(toy._BASE)
    spec = toy.spec(export_rule="cet")
    by_rate = cge.calibrate(sam, spec, export_taxes=ECON.te0, tariffs=ECON.tm0)
    rated = cge.solve_period(by_rate, scenario=cge.Scenario(export_tax=np.zeros(2)))
    np.testing.assert_allclose(rated.exports, named.exports, rtol=1e-10)
    np.testing.assert_allclose(rated.factor_prices, named.factor_prices, rtol=1e-10)


def test_abolishing_it_raises_exports_under_a_cet():
    closure = cge.Closure(foreign="fixed_rate")
    base = cge.solve_period(ECON, closure)
    free = cge.solve_period(ECON, closure, cge.Scenario(export_tax=np.zeros(2)))
    assert free.converged and abs(free.walras) < 1e-12
    assert np.all(free.exports > base.exports)
    assert free.government["export_tax"] == 0.0
    # A rise does the opposite, and its revenue is the rate times exports at the border.
    taxed = cge.solve_period(ECON, closure, cge.Scenario(export_tax=ECON.te0 * 3))
    assert np.all(taxed.exports < base.exports)
    assert taxed.government["export_tax"] == pytest.approx(
        float((ECON.te0 * 3 * taxed.pe * taxed.exports).sum()), rel=1e-12)


def test_the_export_tax_does_not_move_the_tariff_or_the_product_tax():
    closure = cge.Closure(foreign="fixed_rate")
    tariff = cge.solve_period(ECON, closure, cge.Scenario(tariff=np.zeros(2)))
    export = cge.solve_period(ECON, closure, cge.Scenario(export_tax=np.zeros(2)))
    assert tariff.government["export_tax"] == pytest.approx(
        float((ECON.te0 * tariff.pe * tariff.exports).sum()), rel=1e-12)
    assert export.government["tariff"] > 0
    assert abs(tariff.exports.sum() - export.exports.sum()) > 1e-3


@pytest.mark.parametrize("rule", ["cet", "fixed_shares"])
def test_the_accounts_close_under_a_change_in_the_export_tax(rule):
    econ = ECON if rule == "cet" else toy.build_with_export_tax(FIXED)[1]
    period = cge.solve_period(econ, scenario=cge.Scenario(export_tax=econ.te0 * 4))
    acc = cge.national_accounts(period)
    assert abs(acc["gap"]) < 1e-10
    gov = cge.government_account(period)
    assert gov["export_tax"] > 0
    assert gov["revenue"] == pytest.approx(
        sum(gov[k] for k in ("product_tax", "activity_tax", "tariff", "export_tax",
                             "direct_tax", "factor_income")), rel=1e-12)


def test_under_fixed_shares_the_tax_raises_the_border_price():
    econ = toy.build_with_export_tax(FIXED)[1]
    closure = cge.Closure(foreign="fixed_rate")
    base = cge.solve_period(econ, closure)
    taxed = cge.solve_period(econ, closure, cge.Scenario(export_tax=econ.te0 * 4))
    # Exports fetch the output price, so the tax is passed on to the border price.
    ratio = taxed.pe / (taxed.pd)
    np.testing.assert_allclose(ratio, (1 - econ.te0) / (1 - econ.te0 * 4), rtol=1e-12)
    assert np.all(taxed.pe > base.pe)


def test_an_export_tax_that_was_never_calibrated_is_refused():
    _, plain = toy.build()
    with pytest.raises(ValueError, match="export_tax"):
        cge.solve_period(plain, scenario=cge.Scenario(export_tax=np.zeros(2)))


def test_bad_rates_are_refused():
    with pytest.raises(ValueError, match="below 1"):
        cge.solve_period(ECON, scenario=cge.Scenario(export_tax=np.array([1.0, 0.0])))
    with pytest.raises(ValueError, match="length"):
        cge.solve_period(ECON, scenario=cge.Scenario(export_tax=np.zeros(3)))
    sam = toy._frame(toy._BASE)
    with pytest.raises(ValueError, match="exceeds"):
        cge.calibrate(sam, toy.spec(), export_taxes=np.array([0.9, 0.0]))
    with pytest.raises(ValueError, match="not both"):
        cge.calibrate(SAM, toy.spec(export_tax_account="etx"), export_taxes=np.zeros(2))
    with pytest.raises(KeyError, match="not in the SAM"):
        cge.calibrate(toy._frame(toy._BASE), toy.spec(export_tax_account="etx"))


def test_the_replica_refuses_it():
    with pytest.raises(ValueError, match="export tax"):
        cge.to_jump(ECON)


def test_the_regional_preflight_no_longer_lists_it():
    """The regional solver reads export taxes since 0.14 (tests/test_regional_catch_up.py)."""
    from cgelab.regional import _slice_issues
    issues = _slice_issues(dataclasses.replace(ECON, spec=dataclasses.replace(
        ECON.spec, regions=("r1",))), SAM)
    assert not any("export_taxes" in str(i) for i in issues)
