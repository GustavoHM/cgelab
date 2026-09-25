"""Labour-market closures. The numeraire stays a check, and the unemployed are not paid."""

import numpy as np
import pytest

import cgekit as ck
from cgekit.examples import toy


def economy():
    """Capital is the numeraire, so the labour market can be an equation."""
    _, econ = toy.build(toy.spec(numeraire_factor="f_cap"))
    return econ


def test_a_labour_closure_refuses_the_numeraire_factor():
    _, econ = toy.build()  # f_lab is the default numeraire
    with pytest.raises(ValueError, match="numeraire"):
        ck.solve_period(econ, closure=ck.Closure(labour="fixed_real_wage"))


def test_the_wage_curve_refuses_zero_unemployment():
    with pytest.raises(ValueError, match="natural_unemployment"):
        ck.Closure(labour="wage_curve", natural_unemployment=0.0)


@pytest.mark.parametrize("closure", [
    ck.Closure(labour="full_employment"),
    ck.Closure(labour="fixed_real_wage"),
    ck.Closure(labour="elastic_supply", labour_supply_elasticity=0.4),
    ck.Closure(labour="wage_curve", natural_unemployment=0.06, wage_curve_elasticity=0.1),
], ids=["full", "fixed_real", "elastic", "curve"])
def test_the_base_year_still_reproduces(closure):
    econ = economy()
    period = ck.solve_period(econ, closure=closure)
    assert period.converged
    assert abs(period.walras) < 1e-8
    assert np.allclose(period.X, econ.X0, atol=1e-6)
    assert np.allclose(period.Q, econ.Q0, atol=1e-6)
    assert period.real_wage == pytest.approx(1.0, abs=1e-6)
    lab = econ.factor("f_lab")
    assert period.employment == pytest.approx(econ.fac_supply[lab], abs=1e-6)
    if closure.labour == "wage_curve":
        force = econ.fac_supply[lab] / (1.0 - closure.natural_unemployment)
        assert period.unemployment == pytest.approx(force - econ.fac_supply[lab], abs=1e-5)
        assert period.unemployment / force == pytest.approx(closure.natural_unemployment, abs=1e-6)
    else:
        assert period.unemployment == pytest.approx(0.0, abs=1e-6)


def test_a_fixed_real_wage_is_homogeneous_and_creates_unemployment():
    econ = economy()
    base = ck.solve_period(econ, closure=ck.Closure(labour="fixed_real_wage"))
    doubled = ck.solve_period(
        econ, closure=ck.Closure(numeraire=2.0, labour="fixed_real_wage"))
    assert np.allclose(doubled.X, base.X, atol=1e-6)
    assert doubled.factor_prices[econ.factor("f_lab")] == pytest.approx(
        2.0 * base.factor_prices[econ.factor("f_lab")], rel=1e-6)
    assert doubled.real_wage == pytest.approx(base.real_wage, abs=1e-6)

    supply = econ.fac_supply.copy()
    supply[econ.factor("f_lab")] *= 1.10
    shock = ck.Scenario(factor_supply=supply)
    full = ck.solve_period(econ, closure=ck.Closure(labour="full_employment"), scenario=shock)
    fixed = ck.solve_period(econ, closure=ck.Closure(labour="fixed_real_wage"), scenario=shock)
    lab = econ.factor("f_lab")
    assert full.unemployment == pytest.approx(0.0, abs=1e-6)
    assert full.employment == pytest.approx(supply[lab], abs=1e-6)
    # The extra workers do not work and are not paid, so the economy stays at the base year.
    assert fixed.unemployment == pytest.approx(supply[lab] - econ.fac_supply[lab], abs=1e-5)
    assert fixed.employment == pytest.approx(econ.fac_supply[lab], abs=1e-5)
    assert np.allclose(fixed.X, base.X, atol=1e-6)
    assert fixed.hh_income.sum() == pytest.approx(base.hh_income.sum(), rel=1e-6)
    assert abs(fixed.walras) < 1e-8


def test_elastic_supply_answers_a_higher_wage_with_more_employment():
    econ = economy()
    supply = econ.fac_supply.copy()
    supply[econ.factor("f_cap")] *= 1.20
    shock = ck.Scenario(factor_supply=supply)
    full = ck.solve_period(econ, closure=ck.Closure(labour="full_employment"), scenario=shock)
    elastic = ck.solve_period(
        econ, closure=ck.Closure(labour="elastic_supply", labour_supply_elasticity=1.0),
        scenario=shock)
    lab = econ.factor("f_lab")
    assert full.employment == pytest.approx(econ.fac_supply[lab], abs=1e-6)
    assert elastic.employment > full.employment
    assert elastic.real_wage > 1.0
    assert abs(elastic.walras) < 1e-8


def test_the_wage_curve_lets_the_real_wage_fall_when_unemployment_rises():
    econ = economy()
    closure = ck.Closure(labour="wage_curve", natural_unemployment=0.06,
                         wage_curve_elasticity=0.1)
    base = ck.solve_period(econ, closure=closure)
    supply = econ.fac_supply.copy()
    supply[econ.factor("f_lab")] *= 1.10
    shock = ck.solve_period(econ, closure=closure, scenario=ck.Scenario(factor_supply=supply))
    assert shock.unemployment > base.unemployment
    assert shock.real_wage < base.real_wage
    assert abs(shock.walras) < 1e-8


def test_to_jump_exports_the_labour_closure(tmp_path):
    econ = economy()
    _, econ_default = toy.build()
    with pytest.raises(ValueError, match="numeraire"):
        ck.to_jump(econ_default, closure=ck.Closure(labour="fixed_real_wage"),
                   path=tmp_path / "no.jl")
    closure = ck.Closure(labour="fixed_real_wage")
    dados = ck.to_jump(econ, closure=closure, path=tmp_path / "m.jl")
    text = (tmp_path / "m.jl").read_text()
    assert dados["labour"] == "fixed_real_wage"
    assert dados["labour_index"] == econ.factor("f_lab") + 1
    assert "f == NUM && continue" in text
    assert 'LABOUR == "fixed_real_wage"' in text
    assert "paid_supply" in text
    assert "w[LAB] == real_wage_target * P" in text


def test_walras_holds_under_a_tax_in_every_labour_closure():
    econ = economy()
    shock = ck.Scenario(product_tax=econ.ptax * 1.2)
    closures = [
        ck.Closure(labour="full_employment"),
        ck.Closure(labour="fixed_real_wage"),
        ck.Closure(labour="elastic_supply", labour_supply_elasticity=0.4),
        ck.Closure(labour="wage_curve", natural_unemployment=0.06, wage_curve_elasticity=0.1),
    ]
    for closure in closures:
        period = ck.solve_period(econ, closure=closure, scenario=shock)
        assert period.converged
        assert abs(period.walras) < 1e-8, closure.describe()
