"""Regional government closures against the one-region solver."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def labelled_institutions():
    sam, national = toy.build_with_institutions()
    keep = {national.layout.government, national.layout.investment,
            national.layout.rest_of_world}

    def rename(name):
        return name if name in keep else f"br:{name}"

    rows = [(rename(r), rename(c), v) for r, c, v in sam.frame.itertuples(index=False)]
    labelled = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    spec = cge.ModelSpec(
        regions=("br",), numeraire_factor="br:f_lab", factors=("f_lab", "f_cap"),
        sigma_va=national.spec.sigma_va, sigma_armington=national.spec.sigma_armington,
        margins=False)
    return national, cge.calibrate(labelled, spec)


def assert_same(labelled, national, regional):
    assert regional.converged and abs(regional.walras) < 1e-8
    np.testing.assert_allclose(regional.X, national.X, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(regional.Q, national.Q, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(regional.hh_income, national.hh_income, rtol=1e-7, atol=1e-7)
    assert regional.savings == pytest.approx(national.savings, rel=1e-7)
    assert regional.tax_scale == pytest.approx(national.tax_scale, rel=1e-7)
    for key in ("revenue", "consumption", "saving", "direct_tax", "product_tax"):
        assert regional.government[key] == pytest.approx(national.government[key], rel=1e-7)


@pytest.mark.parametrize("rule", ["nominal_fixed", "saving_rate", "spending", "direct", "product"])
def test_government_closures_match_the_national_solver(rule):
    national, labelled = labelled_institutions()
    tfp = np.full(national.n_act, 1.1)
    scenario = cge.Scenario(productivity=tfp)
    if rule in ("nominal_fixed", "saving_rate"):
        closure = cge.Closure(government=rule)
    elif rule == "spending":
        closure = cge.Closure(government_target="saving", government_adjusts="spending")
    elif rule == "direct":
        closure = cge.Closure(government_target="saving", government_adjusts="direct_tax")
        scenario = cge.Scenario(productivity=tfp, product_tax=national.ptax * 0.5)
    else:
        closure = cge.Closure(government_target="revenue", government_adjusts="product_tax")
        scenario = cge.Scenario(productivity=tfp, product_tax=national.ptax * 0.5)
    reference = cge.solve_period(national, closure, scenario)
    result = cge.solve_period(labelled, closure, scenario)
    assert_same(labelled, reference, result)


@pytest.mark.parametrize("rule", ["nominal_fixed", "saving_rate", "spending"])
def test_government_closures_are_homogeneous(rule):
    national, labelled = labelled_institutions()
    closure = (cge.Closure(government=rule, numeraire=2.0) if rule != "spending" else
               cge.Closure(government_target="saving", government_adjusts="spending",
                           numeraire=2.0))
    reference = cge.solve_period(national, closure)
    result = cge.solve_period(labelled, closure)
    assert_same(labelled, reference, result)


def test_saving_rate_uses_consolidated_regional_revenue():
    _, econ = regions.build()
    result = cge.solve_period(econ, cge.Closure(government="saving_rate"))
    assert result.converged and abs(result.walras) < 1e-8
    expected = econ.regional.gov_savrate * result.government["revenue"]
    assert result.government["saving"] == pytest.approx(expected)


def test_deficit_target_met_by_spending():
    _, econ = regions.build()
    closure = cge.Closure(government_target="saving", government_target_value=12.0,
                          government_adjusts="spending")
    result = cge.solve_period(econ, closure)
    assert result.government["saving"] == pytest.approx(12.0)
    assert result.government["consumption"] == pytest.approx(8.0)
    assert abs(result.walras) < 1e-8


def test_zero_tax_levers_are_refused():
    _, econ = regions.build()
    with pytest.raises(ValueError, match="no household pays direct tax"):
        cge.solve_period(econ, cge.Closure(government_target="saving",
                                           government_adjusts="direct_tax"))
    with pytest.raises(ValueError, match="every product tax rate is zero"):
        cge.solve_period(econ, cge.Closure(government_target="revenue",
                                           government_adjusts="product_tax"))


@pytest.mark.parametrize("government", ["real_fixed", "nominal_fixed", "saving_rate"])
@pytest.mark.parametrize("shock", [False, True])
def test_fixed_investment_with_taxes_matches_national_model(government, shock):
    national, regional = labelled_institutions()
    closure = cge.Closure(investment="fixed_real", government=government)
    scenario = (cge.Scenario(productivity=np.full(national.n_act, 1.1),
                             product_tax=national.ptax * 0.5, investment_level=60.0)
                if shock else cge.Scenario())
    expected = cge.solve_period(national, closure, scenario)
    actual = cge.solve_period(regional, closure, scenario)
    assert_same(regional, expected, actual)
    np.testing.assert_allclose(actual.investment_quantity, expected.investment_quantity,
                               rtol=1e-8)
    assert actual.real["inv_real"] == pytest.approx(60.0 if shock else 50.0)
    if not shock:
        np.testing.assert_allclose(actual.X, regional.X0, rtol=1e-8)
