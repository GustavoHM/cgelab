"""What the regional solver now reads, against the one-region national solver."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions, toy

ETA = {"c_agr": 0.6, "c_ind": 1.3}


def one_region(**spec_overrides):
    """The toy economy with institutions, national and as a one-region model."""
    sam, _ = toy.build_with_institutions()
    national = cge.calibrate(sam, toy.spec(**spec_overrides),
                             income_elasticities=ETA if spec_overrides.get(
                                 "household_demand") == "les" else None)
    keep = {national.layout.government, national.layout.investment,
            national.layout.rest_of_world}

    def rename(name):
        return name if name in keep else f"br:{name}"

    rows = [(rename(r), rename(c), v) for r, c, v in sam.frame.itertuples(index=False)]
    labelled = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    spec = cge.ModelSpec(
        regions=("br",), numeraire_factor="br:f_lab", factors=("f_lab", "f_cap"),
        sigma_va=national.spec.sigma_va, sigma_armington=national.spec.sigma_armington,
        margins=False, **spec_overrides)
    elasticities = ({f"br:{c}": v for c, v in ETA.items()}
                    if spec_overrides.get("household_demand") == "les" else None)
    return national, cge.calibrate(labelled, spec, income_elasticities=elasticities)


def assert_same(national, regional):
    assert regional.converged
    assert regional.walras == pytest.approx(national.walras, abs=1e-9)
    np.testing.assert_allclose(regional.X, national.X, rtol=1e-8)
    np.testing.assert_allclose(regional.hh_income, national.hh_income, rtol=1e-8)
    np.testing.assert_allclose(regional.hh_quantity if regional.hh_quantity is not None
                               else 0.0, national.hh_quantity, rtol=1e-8)
    assert regional.savings == pytest.approx(national.savings, rel=1e-8)
    for key in ("revenue", "consumption", "saving", "absorbed"):
        assert regional.government[key] == pytest.approx(national.government[key],
                                                          rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("closure", [cge.Closure(), cge.Closure(government="saving_rate"),
                                     cge.Closure(investment="fixed_real")])
def test_les_households_match_the_national_solver(closure):
    national, regional = one_region(household_demand="les", frisch=-2.0)
    sc = cge.Scenario(productivity=np.full(national.n_act, 1.1))
    assert_same(cge.solve_period(national, closure, sc), cge.solve_period(regional, closure, sc))


@pytest.mark.parametrize("closure", [cge.Closure(), cge.Closure(government="saving_rate")])
def test_government_absorbs_matches_the_national_solver(closure):
    national, regional = one_region()
    sc = cge.Scenario(government_absorbs=3.0)
    a, b = cge.solve_period(national, closure, sc), cge.solve_period(regional, closure, sc)
    assert_same(a, b)
    assert b.government["absorbed"] == 3.0


def test_the_regional_reference_reads_both():
    """The cases validation/replicate_regional.py runs are calibrated and solve."""
    sam, econ = regions.build(taxes=True)
    r = cge.solve_period(econ, scenario=cge.Scenario(government_absorbs=3.0))
    assert r.converged and abs(cge.national_accounts(r)["gap"]) < 1e-9
    eta = {"c": 0.7, "m": 1.0, "g": 1.4}
    _, les = regions.build_with_transport(
        unrouted_good=True, household_demand="les", frisch=-2.0,
        income_elasticities={f"{r}:{g}": v for r in ("r1", "r2") for g, v in eta.items()})
    assert les.regional.hh_beta[0, 2] > 0 and les.regional.hh_committed.sum() > 0
    assert cge.solve_period(les).converged


def one_region_from(sam, national_spec, **regional_spec):
    """Any national toy SAM as a one-region model; tax accounts keep their names."""
    national = cge.calibrate(sam, national_spec)
    keep = {national.layout.government, national.layout.investment,
            national.layout.rest_of_world}
    keep |= {a for a in (national_spec.tariff_account, national_spec.indirect_tax_account,
                         national_spec.export_tax_account) if a}

    def rename(name):
        return name if name in keep else f"br:{name}"

    rows = [(rename(r), rename(c), v) for r, c, v in sam.frame.itertuples(index=False)]
    labelled = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    spec = cge.ModelSpec(
        regions=("br",), numeraire_factor="br:f_lab", factors=("f_lab", "f_cap"),
        sigma_va=national.spec.sigma_va, sigma_armington=national.spec.sigma_armington,
        margins=False, export_rule=national_spec.export_rule,
        tariff_account=national_spec.tariff_account,
        indirect_tax_account=national_spec.indirect_tax_account,
        export_tax_account=national_spec.export_tax_account, **regional_spec)
    return national, cge.calibrate(labelled, spec)


TRADE_CASES = {
    "tariff_and_output_tax": (toy.build_with_trade_taxes,
                              dict(tariff_account="trf", indirect_tax_account="idt"),
                              lambda e: dict(tariff=e.tm0 * 0.0, output_tax=e.tauz0 * 2.0)),
    "tariff_and_output_tax_cet": (toy.build_with_trade_taxes,
                                  dict(tariff_account="trf", indirect_tax_account="idt",
                                       export_rule="cet"),
                                  lambda e: dict(tariff=e.tm0 * 0.0, output_tax=e.tauz0 * 2.0)),
    "export_tax_cet": (toy.build_with_export_tax,
                       dict(tariff_account="trf", export_tax_account="etx", export_rule="cet"),
                       lambda e: dict(export_tax=e.te0 * 3.0, tariff=e.tm0 * 0.5)),
    "export_tax_fixed": (toy.build_with_export_tax,
                         dict(tariff_account="trf", export_tax_account="etx"),
                         lambda e: dict(export_tax=e.te0 * 3.0)),
}


@pytest.mark.parametrize("name", list(TRADE_CASES))
@pytest.mark.parametrize("closure", [cge.Closure(), cge.Closure(government="saving_rate")])
def test_trade_and_output_taxes_match_the_national_solver(name, closure):
    build, spec_kw, shock = TRADE_CASES[name]
    sam, _ = build()
    national, regional = one_region_from(sam, toy.spec(**spec_kw))
    nat_sc = cge.Scenario(world_price=np.array([1.1, 0.95]), **shock(national))
    reg_kw = shock(national)
    reg_sc = cge.Scenario(world_price=np.array([1.1, 0.95]), **reg_kw)
    a, b = cge.solve_period(national, closure, nat_sc), cge.solve_period(regional, closure, reg_sc)
    assert_same(a, b)
    np.testing.assert_allclose(b.exports, a.exports, rtol=1e-8)
    np.testing.assert_allclose(b.imports, a.imports, rtol=1e-8)
    for key in ("tariff", "output_tax", "export_tax"):
        assert b.government[key] == pytest.approx(a.government[key], rel=1e-8, abs=1e-10)


def test_regional_trade_taxes_are_refused_where_they_would_not_hold():
    sam, _ = regions.build(taxes=True)
    with pytest.raises(ValueError, match="differ across the origins|product tax across regions"):
        cge.calibrate(sam, regions.spec(), tariffs=np.array([0.05, 0.02]))
    econ = cge.calibrate(sam, regions.spec(), tariffs=np.array([0.05, 0.05]))
    with pytest.raises(ValueError, match="one rate per local commodity"):
        cge.solve_period(econ, scenario=cge.Scenario(tariff=np.zeros(2)))
    _, plain = regions.build(taxes=True)
    with pytest.raises(ValueError, match="calibrated"):
        cge.solve_period(plain, scenario=cge.Scenario(tariff=np.zeros(1)))
