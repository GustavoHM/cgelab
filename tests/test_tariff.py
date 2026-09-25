"""A tariff is not a product tax, and an indirect tax on output is neither of them."""

import numpy as np
import pytest

import cgekit as ck
from cgekit.examples import toy


def _with_tariff(rate=0.05):
    sam, plain = toy.build()
    tm = np.full(plain.n_com, rate)
    econ = ck.calibrate(sam, toy.spec(), tariffs=tm)
    return plain, econ, tm


def test_a_calibrated_tariff_still_reproduces_the_base_year():
    plain, econ, tm = _with_tariff()
    plain_period = ck.solve_period(plain)
    period = ck.solve_period(econ)
    assert period.converged and abs(period.walras) < 1e-8
    # The reference rate does not move the base year. The product tax still
    # closes it; the tariff is collected when the rate changes.
    assert np.allclose(period.Q, plain_period.Q, atol=1e-8)
    assert np.allclose(period.imports, plain_period.imports, atol=1e-8)
    assert np.allclose(econ.ptax, plain.ptax)


def test_raising_the_tariff_cuts_imports_and_keeps_walras():
    # The exchange rate is held so the relative-price effect is visible. With
    # foreign saving fixed, the currency can appreciate by more than the wedge
    # and import volumes can rise.
    _, econ, tm = _with_tariff()
    closure = ck.Closure(foreign="fixed_rate")
    base = ck.solve_period(econ, closure=closure)
    shock = ck.solve_period(econ, closure=closure, scenario=ck.Scenario(tariff=tm * 3))
    assert shock.converged and abs(shock.walras) < 1e-8
    assert shock.imports.sum() < base.imports.sum()


def test_a_tariff_with_a_fixed_current_account_appreciates_the_currency():
    _, econ, tm = _with_tariff()
    base = ck.solve_period(econ)
    shock = ck.solve_period(econ, scenario=ck.Scenario(tariff=tm * 3))
    assert shock.converged and abs(shock.walras) < 1e-8
    assert shock.exchange_rate < base.exchange_rate


def test_the_product_tax_does_not_move_the_tariff():
    _, econ, tm = _with_tariff()
    closure = ck.Closure(foreign="fixed_rate")
    base = ck.solve_period(econ, closure=closure)
    tax = ck.solve_period(econ, closure=closure,
                          scenario=ck.Scenario(product_tax=econ.ptax * 1.5))
    duty = ck.solve_period(econ, closure=closure,
                           scenario=ck.Scenario(tariff=np.zeros_like(tm)))
    assert abs(tax.walras) < 1e-8 and abs(duty.walras) < 1e-8
    # Abolishing the tariff is a different experiment from raising the product tax.
    assert abs(duty.imports.sum() - tax.imports.sum()) > 1e-4
    assert duty.imports.sum() > base.imports.sum()


def test_a_tariff_that_was_never_calibrated_is_refused():
    _, econ = toy.build()
    with pytest.raises(ValueError, match="tariff"):
        ck.solve_period(econ, scenario=ck.Scenario(tariff=np.zeros(econ.n_com)))


def test_tariff_revenue_cannot_exceed_the_commodity_tax():
    sam, econ = toy.build()
    with pytest.raises(ValueError, match="exceeds"):
        ck.calibrate(sam, toy.spec(), tariffs=np.full(econ.n_com, 5.0))


def _named_tax_sam(account, take):
    """Move `take` of each commodity's product tax onto its own account."""
    from cgekit.examples.toy import _BASE, _frame

    entries = []
    total = 0.0
    for row, col, value in _BASE:
        if row == "gov" and col in take:
            entries.append(("gov", col, value - take[col]))
            entries.append((account, col, take[col]))
            total += take[col]
        else:
            entries.append((row, col, value))
    entries.append(("gov", account, total))
    return _frame(entries)


def test_a_named_tariff_account_keeps_the_base_year():
    from cgekit.examples.toy import _frame, _BASE, spec

    sam = _named_tax_sam("trf", {"c_agr": 1.0, "c_ind": 2.0})
    plain = ck.calibrate(_frame(_BASE), spec())
    econ = ck.calibrate(sam, spec(tariff_account="trf"))
    base = ck.solve_period(plain)
    period = ck.solve_period(econ)
    assert period.converged and abs(period.walras) < 1e-8
    assert np.allclose(period.Q, base.Q, atol=1e-8)
    assert np.allclose(period.imports, base.imports, atol=1e-8)
    assert np.allclose(econ.ptax, plain.ptax)
    assert np.allclose(econ.tm0, [1.0 / 15.0, 2.0 / 25.0])
    assert econ.tauz0 is None
    assert not any("trf" in (a, b) for a, b, _ in econ.diagnostics["unmodelled_flows"])


def test_a_named_output_tax_keeps_the_base_year_and_a_rise_shifts_demand_to_imports():
    from cgekit.examples.toy import spec

    sam = _named_tax_sam("idt", {"c_agr": 2.0, "c_ind": 4.0})
    econ = ck.calibrate(sam, spec(indirect_tax_account="idt"))
    assert np.allclose(econ.tauz0, [2.0 / 100.0, 4.0 / 200.0])
    assert econ.tm0 is None
    closure = ck.Closure(foreign="fixed_rate")
    base = ck.solve_period(econ, closure=closure)
    shock = ck.solve_period(
        econ, closure=closure,
        scenario=ck.Scenario(output_tax=econ.tauz0 * 3))
    assert base.converged and shock.converged
    assert abs(base.walras) < 1e-8 and abs(shock.walras) < 1e-8
    assert np.allclose(base.Q, [100.0, 200.0], atol=1e-6)
    assert shock.imports.sum() > base.imports.sum()


def test_an_output_tax_rate_reproduces_the_base_year():
    sam, plain = toy.build()
    rate = np.full(plain.n_com, 0.02)
    econ = ck.calibrate(sam, toy.spec(), output_taxes=rate)
    plain_period = ck.solve_period(plain)
    period = ck.solve_period(econ)
    assert period.converged and abs(period.walras) < 1e-8
    assert np.allclose(period.Q, plain_period.Q, atol=1e-8)
    assert np.allclose(econ.tauz0, rate)
    closure = ck.Closure(foreign="fixed_rate")
    abolished = ck.solve_period(
        econ, closure=closure, scenario=ck.Scenario(output_tax=np.zeros_like(rate)))
    assert abs(abolished.walras) < 1e-8
    assert abolished.imports.sum() < ck.solve_period(econ, closure=closure).imports.sum()


def test_an_output_tax_that_was_never_calibrated_is_refused():
    _, econ = toy.build()
    with pytest.raises(ValueError, match="output_tax"):
        ck.solve_period(econ, scenario=ck.Scenario(output_tax=np.zeros(econ.n_com)))


def test_output_tax_revenue_cannot_exceed_the_commodity_tax():
    sam, econ = toy.build()
    with pytest.raises(ValueError, match="exceeds"):
        ck.calibrate(sam, toy.spec(), output_taxes=np.full(econ.n_com, 0.5))


def test_to_jump_writes_the_tariff_power(tmp_path):
    _, econ, tm = _with_tariff()
    dados = ck.to_jump(econ, scenario=ck.Scenario(tariff=tm * 2), path=tmp_path / "m.jl")
    text = (tmp_path / "m.jl").read_text()
    assert np.allclose(dados["tm"], tm * 2)
    assert np.allclose(dados["tm0"], tm)
    assert "(1 + tm[c]) / (1 + tm0[c])" in text
    assert "pdom[c] * dsh[c] * armsh[c] * Aabs[c]" in text
