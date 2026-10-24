"""The one-call report composes the public robustness tools without changing them."""

import builtins

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.robustness_report import _distributions, _make_figures, _one_at_a_time, _range_table

SEED = 11
BLOCKS = {"intermediate": (["c_agr", "c_ind"], ["a_agr", "a_ind"])}
SAM_KWARGS = {"balance_kwargs": {"rel_tol": 1e-12, "abs_tol": 1e-10}}
SHOCK = cge.Scenario(world_price=np.array([1.0, 1.2]))


def measure(base, alt):
    return {"gdp_pct": 100 * (cge.real_gdp(alt)["expenditure"]
                              / cge.real_gdp(base)["expenditure"] - 1)}


def distributions(report):
    return {name: cge.Triangular(row.low, row["mode"], row.high)
            for name, row in report.ranges.iterrows()}


@pytest.fixture(scope="module")
def report():
    sam, _ = toy.build()
    return cge.robustness_report(sam, toy.spec(), measure=measure, scenario=SHOCK,
                                 sam_blocks=BLOCKS, draws=4, seed=SEED,
                                 sam_kwargs=SAM_KWARGS)


def test_default_ranges_tables_and_refused_oat_point(report):
    assert set(report.tables) == {"ranges", "oat", "oat_failed", "quadrature",
                                  "monte_carlo", "closures", "sam", "money"}
    assert list(report.ranges.index) == ["sigma_va", "sigma_armington"]
    assert report.ranges.loc["sigma_va", "low"] == pytest.approx(0.4)
    assert report.ranges.loc["sigma_va", "high"] == pytest.approx(1.2)
    assert report.ranges.loc["sigma_armington", "mode"] == pytest.approx(2.0)
    assert report.ranges.source.str.contains(r"Rose et al. \(2023\)", regex=True).all()
    assert "triangular shape is a package convention" in report.ranges.iloc[0].source
    failed = report.tables["oat_failed"]
    assert list(failed[["parameter", "point"]].itertuples(index=False, name=None)) == [
        ("sigma_armington", "low")]
    assert "exchange rate is not determined" in failed.iloc[0].reason
    assert np.isnan(report.tables["oat"].loc[("gdp_pct", "sigma_armington"), "low"])
    assert report.figures == {}


def test_report_sections_equal_the_direct_public_calls(report):
    sam, _ = toy.build()
    params = distributions(report)
    run = cge.spec_sweep(sam, toy.spec(), measure=measure, scenario=SHOCK)
    direct_mc = cge.sensitivity(run, params, method="monte_carlo", draws=4, seed=SEED)
    pd.testing.assert_frame_equal(report.parts["monte_carlo"].points, direct_mc.points)
    direct_quad = cge.sensitivity(run, params, seed=SEED)
    pd.testing.assert_frame_equal(report.parts["quadrature"].points, direct_quad.points)

    drawn = cge.sam_draws(sam, blocks=BLOCKS, draws=4, seed=SEED, **SAM_KWARGS)
    direct_sam = cge.sam_sweep(drawn, toy.spec(), measure=measure, scenario=SHOCK,
                               parameters=params)
    pd.testing.assert_frame_equal(report.parts["sam"].points, direct_sam.points)

    central = cge.calibrate(sam, toy.spec())
    grid = cge.closure_grid(investment=["savings_driven", "fixed_real"],
                            foreign=["fixed_saving", "fixed_rate"])
    direct_closures = cge.closure_sweep(central, grid, scenario=SHOCK, measure=measure)
    pd.testing.assert_frame_equal(report.parts["closures"].table, direct_closures.table)
    base = cge.solve_period(central)
    alt = cge.solve_period(central, scenario=SHOCK, warm_start=base)
    pd.testing.assert_frame_equal(report.tables["money"], cge.in_money(base, alt))


def test_oat_central_is_the_central_model_and_extrema_are_computed(report):
    sam, _ = toy.build()
    run = cge.spec_sweep(sam, toy.spec(), measure=measure, scenario=SHOCK)
    central = run(report.ranges["mode"].to_dict())["gdp_pct"]
    table = report.tables["oat"]
    np.testing.assert_allclose(table["central"], central, rtol=1e-12)
    for _, row in table.iterrows():
        values = row[["low", "central", "high"]].dropna()
        assert row["min"] == values.min() and row["max"] == values.max()
        assert row["spread"] == pytest.approx(values.max() - values.min())


def test_unsourced_active_elasticities_need_user_values():
    spec = toy.spec(energy=["c_ind"])
    with pytest.raises(ValueError, match="sigma_vae"):
        _distributions(spec, "rose2023", None)
    params = {"sigma_vae": cge.Uniform(0.2, 0.8),
              "sigma_energy": cge.Uniform(0.5, 1.5)}
    distributions, sources = _distributions(spec, "rose2023", params)
    assert set(distributions) == {"sigma_va", "sigma_armington", "sigma_vae", "sigma_energy"}
    assert sources["sigma_vae"] == "user distribution"
    with pytest.raises(KeyError, match="sigma_foo"):
        _distributions(toy.spec(), "rose2023", {"sigma_foo": cge.Uniform(1, 2)})


def test_arndt_supports_and_user_ranges_are_explicit():
    params = {"sigma_va": cge.Triangular(0.2, 0.8, 1.4)}
    distributions, sources = _distributions(toy.spec(export_rule="cet"), "arndt2002", params)
    assert distributions["sigma_armington"].high == 9.0
    assert distributions["sigma_cet"].high == 5.0
    assert distributions["sigma_armington"].mode == 1.5
    assert sources["sigma_cet"].startswith("Arndt, Robinson and Tarp")
    custom, sources = _distributions(toy.spec(), {"sigma_va": (0.2, 0.5, 1.0),
                                                 "sigma_armington": (1.0, 2.0, 4.0)}, None)
    table = _range_table(custom, sources)
    assert table.loc["sigma_va", "mode"] == 0.5
    assert table.source.str.contains("user range").all()


def test_asymmetric_support_skips_quadrature_with_a_reason():
    sam, _ = toy.build()
    report = cge.robustness_report(
        sam, toy.spec(export_rule="cet"), measure=measure, scenario=SHOCK,
        sam_blocks=BLOCKS, ranges="arndt2002",
        parameters={"sigma_va": cge.Triangular(0.2, 0.8, 1.4)},
        draws=2, seed=SEED, sam_kwargs=SAM_KWARGS)
    assert report.parts["quadrature"] is None
    assert report.tables["quadrature"].empty
    assert "symmetric" in report.tables["quadrature"].attrs["skipped"]
    assert not report.tables["monte_carlo"].empty


def test_oat_accepts_a_scalar_result():
    ranges = pd.DataFrame({"low": [0.0], "mode": [1.0], "high": [2.0],
                           "source": ["user"]}, index=["x"])
    table, failed = _one_at_a_time(lambda values: 2 * values["x"], ranges, ["result"])
    assert list(table.loc["result", ["low", "central", "high"]].iloc[0]) == [0.0, 2.0, 4.0]
    assert failed.empty


def test_sobol_is_optional_and_matches_the_direct_call():
    sam, _ = toy.build()
    report = cge.robustness_report(sam, toy.spec(), measure=measure, scenario=SHOCK,
                                   sam_blocks=BLOCKS, draws=2, seed=SEED, sobol=True,
                                   sobol_n=4, sam_kwargs=SAM_KWARGS)
    assert len(report.parts["sobol"].points) == 4 * (2 + 2)
    direct = cge.sobol(cge.spec_sweep(sam, toy.spec(), measure=measure, scenario=SHOCK),
                       distributions(report), n=4, seed=SEED)
    pd.testing.assert_frame_equal(report.tables["sobol"], direct.indices())


def test_plot_dependency_has_an_actionable_error(report, monkeypatch):
    original_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with pytest.raises(ImportError, match=r"cgelab\[plot\]"):
        _make_figures(report)


def test_figures_when_matplotlib_is_installed(report):
    pytest.importorskip("matplotlib")
    figures = _make_figures(report)
    assert {"tornado_gdp_pct", "distribution_gdp_pct"} <= set(figures)
    import matplotlib.pyplot as plt
    plt.close("all")
