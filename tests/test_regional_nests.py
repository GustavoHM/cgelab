"""Margins and CET exports in the regional origin nest."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def labelled_margins(export_rule="fixed_shares"):
    sam, national = toy.build_with_margins(toy.spec(export_rule=export_rule, sigma_cet=3.0))
    keep = {national.layout.government, national.layout.investment,
            national.layout.rest_of_world}

    def rename(name):
        return name if name in keep else f"br:{name}"

    rows = [(rename(r), rename(c), v) for r, c, v in sam.frame.itertuples(index=False)]
    labelled = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    spec = cge.ModelSpec(
        regions=("br",), numeraire_factor="br:f_lab", factors=("f_lab", "f_cap"),
        sigma_va=national.spec.sigma_va, sigma_armington=national.spec.sigma_armington,
        margins=True, export_rule=export_rule, sigma_cet=3.0)
    return national, cge.calibrate(labelled, spec)


def assert_same_solution(national, regional, labelled):
    assert regional.converged and abs(regional.walras) < 1e-8
    for name in national.econ.acts:
        assert regional.X[labelled.activity(f"br:{name}")] == pytest.approx(
            national.X[national.econ.activity(name)], rel=1e-8, abs=1e-8)
    for name in national.econ.coms:
        i, j = national.econ.commodity(name), labelled.commodity(f"br:{name}")
        assert regional.Q[j] == pytest.approx(national.Q[i], rel=1e-8, abs=1e-8)
        assert regional.exports[j] == pytest.approx(national.exports[i], rel=1e-8, abs=1e-8)
        assert regional.imports[j] == pytest.approx(national.imports[i], rel=1e-8, abs=1e-8)


def test_one_region_margins_match_the_national_solver():
    national, labelled = labelled_margins()
    base = cge.solve_period(national)
    result = cge.solve_period(labelled)
    assert np.any(labelled.regional.mq != 0)
    assert_same_solution(base, result, labelled)


def test_margins_still_match_after_a_productivity_shock():
    national, labelled = labelled_margins()
    tfp = np.full(national.n_act, 1.1)
    base = cge.solve_period(national, scenario=cge.Scenario(productivity=tfp))
    result = cge.solve_period(labelled, scenario=cge.Scenario(productivity=tfp))
    assert_same_solution(base, result, labelled)


def test_one_region_cet_matches_the_national_solver_after_price_shocks():
    national, labelled = labelled_margins(export_rule="cet")
    scenario = cge.Scenario(world_price=np.array([1.1, 0.95, 1.0]),
                            export_price=np.array([1.2, 1.05, 1.0]))
    base = cge.solve_period(national, scenario=scenario)
    result = cge.solve_period(labelled, scenario=scenario)
    assert_same_solution(base, result, labelled)


def test_two_region_margin_sam_reproduces_every_activity():
    entries = list(regions.ENTRIES)
    for region in ("r1", "r2"):
        entries += [
            (f"{region}:f_lab", f"{region}:a_m", 8.0),
            (f"{region}:f_cap", f"{region}:a_m", 4.0),
            (f"{region}:a_m", f"{region}:m", 12.0),
            (f"{region}:m", f"{region}:c", 12.0),
            (f"{region}:m", f"{region}:m", -12.0),
            (f"{region}:c", f"{region}:hh", 12.0),
            (f"{region}:hh", f"{region}:f_lab", 8.0),
            (f"{region}:hh", f"{region}:f_cap", 4.0),
        ]
    sam = cge.SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]))
    spec = regions.spec(
        activities=[f"{r}:{a}" for r in ("r1", "r2") for a in ("a", "a_m")],
        commodities=[f"{r}:{c}" for r in ("r1", "r2") for c in ("c", "m")],
        margins=True)
    econ = cge.calibrate(sam, spec)
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-8
    np.testing.assert_allclose(result.X, [100.0, 12.0, 100.0, 12.0], atol=1e-8)
    assert np.count_nonzero(econ.regional.mq) == 1


def test_two_region_cet_reproduces_and_responds_to_export_prices():
    _, econ = regions.build(regions.spec(export_rule="cet", sigma_cet=3.0))
    base = cge.solve_period(econ)
    shocked = cge.solve_period(econ, scenario=cge.Scenario(export_price=[1.1]), warm_start=base)
    np.testing.assert_allclose(base.Q, [100.0, 100.0], atol=1e-8)
    np.testing.assert_allclose(base.exports, [10.0, 10.0], atol=1e-8)
    assert (shocked.exports > base.exports).all()
    assert abs(shocked.walras) < 1e-8


def test_regional_export_price_validation():
    _, fixed = regions.build()
    with pytest.raises(ValueError, match="export_rule='cet'"):
        cge.solve_period(fixed, scenario=cge.Scenario(export_price=[1.1]))
    _, cet = regions.build(regions.spec(export_rule="cet"))
    with pytest.raises(ValueError, match="one foreign price"):
        cge.solve_period(cet, scenario=cge.Scenario(export_price=[1.1, 1.0]))


def test_price_iteration_failure_is_not_accepted(monkeypatch):
    from cgelab.regional import _prices

    _, econ = labelled_margins()
    monkeypatch.setattr("cgelab.regional._PRICE_ITERS", 1)
    with pytest.raises(RuntimeError, match="price fixed point did not converge"):
        _prices(econ, econ.regional, np.ones(econ.n_fac), 1.0,
                np.ones(econ.regional.n_local), np.ones(econ.regional.n_local),
                np.ones(econ.n_act), 1.0)
