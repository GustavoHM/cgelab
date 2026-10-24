"""Robustness to the closure, and results in money.

The sweep is checked against the same solves done by hand, and each closure
against what it says it holds. The money table is checked against the
functions it is built from: ``national_accounts`` at current prices,
``real_gdp`` and the Laspeyres volume of ``price_indices`` at base-year prices,
``equivalent_variation`` for households; and against homogeneity, since a
doubled numeraire doubles every current-price level and leaves the base-price
ones where they were.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy

SHOCK = cge.Scenario(world_price=np.array([1.0, 1.2]))


def econ():
    return toy.build()[1]


def gdp_pct(base, alt):
    return 100.0 * (cge.real_gdp(alt)["expenditure"] / cge.real_gdp(base)["expenditure"] - 1.0)


def test_one_closure_is_the_solve_done_by_hand():
    e = econ()
    closure = cge.Closure(foreign="fixed_rate")
    sweep = cge.closure_sweep(e, {"fixed rate": closure}, scenario=SHOCK, measure=gdp_pct)
    base = cge.solve_period(e, closure)
    alt = cge.solve_period(e, closure, SHOCK, warm_start=base)
    assert sweep.table.loc["fixed rate", "result"] == gdp_pct(base, alt)
    assert sweep.converged["fixed rate"]
    assert sweep.walras["fixed rate"] < 1e-9
    assert sweep.failed == {}


def test_each_closure_holds_what_it_says_it_holds():
    e = econ()

    def measure(base, alt):
        return {"inv_real": alt.real["inv_real"] / base.real["inv_real"] - 1.0,
                "er": alt.exchange_rate, "real_wage": alt.real_wage / base.real_wage - 1.0}

    closures = {"investment": cge.Closure(investment="fixed_real"),
                "rate": cge.Closure(foreign="fixed_rate", exchange_rate=1.0),
                "wage": cge.Closure(labour="fixed_real_wage")}
    # Labour is not the numeraire, so its wage can be pinned.
    sweep = cge.closure_sweep(toy.build(toy.spec(numeraire_factor="f_cap"))[1], closures,
                              scenario=SHOCK, measure=measure)
    t = sweep.table
    assert abs(t.loc["investment", "inv_real"]) < 1e-10
    assert abs(t.loc["rate", "er"] - 1.0) < 1e-12
    assert abs(t.loc["wage", "real_wage"]) < 1e-10
    assert sweep.converged.all()


def test_a_grid_of_closures_and_the_combinations_refused():
    grid = cge.closure_grid(investment=["savings_driven", "fixed_real"],
                            foreign=["fixed_saving", "fixed_rate"],
                            labour=["full_employment", "wage_curve"])
    assert len(grid) == 8
    assert "default" in grid and "investment=fixed_real, foreign=fixed_rate" in grid
    sweep = cge.closure_sweep(econ(), grid, scenario=SHOCK, measure=gdp_pct)
    # Under a wage curve the base year needs a rate of unemployment, which the
    # grid did not give: those four are refused by Closure, with its reason.
    refused = [k for k in grid if "wage_curve" in k]
    assert sorted(sweep.failed) == sorted(refused)
    assert all("natural_unemployment" in v for v in sweep.failed.values())
    assert sweep.table.loc[refused].isna().all().all()
    assert not sweep.converged[refused].any()
    assert sweep.converged.drop(refused).all()
    # Given one, they solve.
    grid = cge.closure_grid(labour=["full_employment", "wage_curve"], natural_unemployment=0.08)
    capital_numeraire = toy.build(toy.spec(numeraire_factor="f_cap"))[1]
    sweep = cge.closure_sweep(capital_numeraire, grid, scenario=SHOCK, measure=gdp_pct)
    assert sweep.failed == {} and len(sweep.table) == 2


def test_the_range_over_closures():
    grid = cge.closure_grid(investment=["savings_driven", "fixed_real"],
                            foreign=["fixed_saving", "fixed_rate"])
    sweep = cge.closure_sweep(econ(), grid, scenario=SHOCK,
                              measure=lambda b, a: {"gdp": gdp_pct(b, a), "er": a.exchange_rate})
    r = sweep.range()
    t = sweep.table
    assert r.loc["gdp", "min"] == t["gdp"].min() and r.loc["gdp", "max"] == t["gdp"].max()
    assert r.loc["gdp", "argmin"] == t["gdp"].idxmin()
    assert r.loc["gdp", "spread"] == pytest.approx(t["gdp"].max() - t["gdp"].min())
    assert bool(r.loc["gdp", "same_sign"]) and bool(r.loc["er", "same_sign"])


def test_a_list_is_labelled_by_what_differs_from_the_default():
    sweep = cge.closure_sweep(econ(), [cge.Closure(), {"investment": "fixed_real"}],
                              scenario=SHOCK, measure=gdp_pct)
    assert list(sweep.table.index) == ["default", "investment=fixed_real"]
    with pytest.raises(ValueError, match="twice"):
        cge.closure_sweep(econ(), [cge.Closure(), cge.Closure()], scenario=SHOCK,
                          measure=gdp_pct)
    with pytest.raises(ValueError, match="no closures"):
        cge.closure_sweep(econ(), {}, scenario=SHOCK, measure=gdp_pct)
    with pytest.raises(TypeError, match="not Closure fields"):
        cge.closure_grid(investmnet=["fixed_real"])


def test_a_sweep_of_paths():
    years = list(range(2021, 2026))
    growth = cge.Growth(labour=0.01)

    def policy(year, t):
        return SHOCK if t >= 2 else cge.Scenario()

    def measure(ref, alt):
        return {y: gdp_pct(r, a) for y, r, a in zip(ref.years, ref.periods, alt.periods)}

    grid = cge.closure_grid(investment=["savings_driven", "fixed_real"])
    sweep = cge.closure_sweep(econ(), grid, scenario=policy, measure=measure, years=years,
                              growth=growth)
    ref = cge.run_path(econ(), cge.Closure(investment="fixed_real"), years=years, growth=growth)
    alt = cge.run_path(econ(), cge.Closure(investment="fixed_real"), years=years, growth=growth,
                       scenario=policy)
    np.testing.assert_allclose(sweep.table.loc["investment=fixed_real"].to_numpy(),
                               list(measure(ref, alt).values()), rtol=1e-12)
    assert (sweep.table[[2021, 2022]].abs() < 1e-9).all().all()


# ---------------------------------------------------------------------------
# Results in money
# ---------------------------------------------------------------------------

def solved(closure=None):
    e = econ()
    base = cge.solve_period(e, closure)
    return base, cge.solve_period(e, closure, SHOCK, warm_start=base)


def test_no_change_is_zero_and_the_base_year_is_the_sam():
    base, _ = solved()
    m = cge.in_money(base, base)
    assert (m[["change", "change_base_prices", "pct_base_prices"]].fillna(0) == 0).all().all()
    fd = m.loc["final_demand"]
    np.testing.assert_allclose(fd.loc[["consumption", "government", "investment", "exports",
                                       "imports", "gdp"], "reference"],
                               [135.0, 20.0, 50.0, 30.0, 40.0, 195.0], rtol=1e-12)
    # At the base year current and base-year prices are the same thing.
    np.testing.assert_allclose(m["reference"], m["reference_base_prices"], rtol=1e-12)
    np.testing.assert_allclose(m.loc["factor_income", "reference"], [110.0, 60.0], rtol=1e-12)


def test_levels_agree_with_the_functions_they_come_from():
    base, alt = solved()
    m = cge.in_money(base, alt)
    na = cge.national_accounts(alt)
    for k in ("consumption", "government", "investment", "exports", "imports"):
        assert m.loc[("final_demand", k), "alternative"] == pytest.approx(na[k], rel=1e-12)
    assert m.loc[("final_demand", "gdp"), "alternative"] == pytest.approx(
        na["gdp_expenditure"], rel=1e-12)
    assert m.loc[("final_demand", "gdp"), "alternative_base_prices"] == pytest.approx(
        cge.real_gdp(alt)["expenditure"], rel=1e-12)
    volume = cge.price_indices(alt)["quantity_laspeyres"]
    for k in ("consumption", "investment", "exports", "imports"):
        assert 1 + m.loc[("final_demand", k), "pct_base_prices"] / 100 == pytest.approx(
            volume[k], rel=1e-12)
    assert m.loc["factor_income", "alternative"].sum() == pytest.approx(na["factor_income"],
                                                                        rel=1e-12)
    g = cge.government_account(alt)
    assert m.loc[("government", "revenue"), "alternative"] == pytest.approx(g["revenue"])


def test_household_change_at_base_prices_is_the_ev_at_base_prices():
    e = econ()
    years = [2021, 2022]
    ref = cge.run_path(e, years=years)
    alt = cge.run_path(e, years=years, scenario=SHOCK)
    m = cge.in_money(ref, alt)
    # equivalent_variation leaves the base-year price index to the caller; at
    # the base year it is each household's index, which a solve of it gives.
    p0 = cge.solve_period(e).real["hh_price_index"]
    ev = cge.equivalent_variation(ref, alt).set_index(["year", "account"])["ev_base"]
    for y in years:
        for i, h in enumerate(e.hhs):
            assert m.loc[(y, "households", h), "change_base_prices"] == pytest.approx(
                ev[(y, h)] * p0[i], rel=1e-10, abs=1e-12)
    assert list(m.index.get_level_values("year").unique()) == years
    assert list(cge.in_money(ref, alt, years=[2022]).index.get_level_values(0).unique()) == [2022]


def test_a_doubled_numeraire_doubles_current_prices_only():
    one = cge.in_money(*solved())
    two = cge.in_money(*solved(cge.Closure(numeraire=2.0)))
    current = ["reference", "alternative", "change"]
    np.testing.assert_allclose(two[current], 2.0 * one[current], rtol=1e-9, atol=1e-12)
    base = ["reference_base_prices", "alternative_base_prices", "change_base_prices",
            "pct_base_prices"]
    np.testing.assert_allclose(two[base].fillna(0), one[base].fillna(0), rtol=1e-9, atol=1e-10)


def test_the_unit_and_what_in_money_refuses():
    base, alt = solved()
    one = cge.in_money(base, alt)
    bn = cge.in_money(base, alt, unit=1e-3, label="billions")
    assert bn.attrs["unit"] == "billions"
    levels = [c for c in one.columns if c != "pct_base_prices"]
    np.testing.assert_allclose(bn[levels], 1e-3 * one[levels], rtol=1e-14)
    np.testing.assert_allclose(bn["pct_base_prices"].fillna(0), one["pct_base_prices"].fillna(0))
    with pytest.raises(ValueError, match="unit"):
        cge.in_money(base, alt, unit=0.0)
    path = cge.run_path(econ(), years=[2021])
    with pytest.raises(TypeError, match="both"):
        cge.in_money(base, path)
    with pytest.raises(ValueError, match="paths"):
        cge.in_money(base, alt, years=[2021])
    with pytest.raises(ValueError, match="2030"):
        cge.in_money(path, path, years=[2030])
