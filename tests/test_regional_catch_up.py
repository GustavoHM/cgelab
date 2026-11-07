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
