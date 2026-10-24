"""Putty-clay capital: an old vintage fixed in its sector, a new one that moves.

The checks come from other code paths and from arithmetic done by hand. With
the same elasticity in both bundles and between them, the two-bundle value
added is one CES over the same factors, which the model without vintages
solves. Under fixed proportions ex post, the old vintage's value added is its
capital over the capital share, whatever the prices. A one-good economy whose
new capital is its investment sits on its steady path, and as the elasticity
between the vintages grows, it approaches the economy with one capital stock at
a rate of one over that elasticity.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.examples.toy import _BASE, _frame

SHOCK = cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.0, 1.2]))
FACTORS = ("f_lab", "f_cap", "f_cap_new")
YEARS = list(range(2021, 2041))


def two_sector(share=0.1, **spec):
    sam = cge.split_vintages(_frame(_BASE), share)
    return cge.calibrate(sam, toy.spec(factors=FACTORS, vintages={"f_cap": "f_cap_new"},
                                       **spec))


ONE_GOOD = dict(activities=["a_x"], commodities=["c_x"], households=["hh"], government="gov",
                investment="s_i", rest_of_world="row", margins=False,
                numeraire_factor="f_lab")


def one_good(sigma=0.8, top=20.0, share=0.05):
    sam, _ = toy.build_one_good()
    return cge.calibrate(cge.split_vintages(sam, share), cge.ModelSpec(
        factors=FACTORS, vintages={"f_cap": "f_cap_new"}, sigma_va=sigma,
        vintage_sigma=sigma, vintage_substitution=top, **ONE_GOOD))


def test_split_vintages_adds_back_to_the_capital_account():
    sam = _frame(_BASE)
    split = cge.split_vintages(sam, {"a_agr": 0.2, "a_ind": 0.1})
    m, s = sam.matrix(), split.matrix()
    np.testing.assert_allclose(s.loc["f_cap_new", ["a_agr", "a_ind"]], [5.0, 3.5])
    np.testing.assert_allclose(s.loc["f_cap", ["a_agr", "a_ind"]] + s.loc["f_cap_new",
                               ["a_agr", "a_ind"]], m.loc["f_cap", ["a_agr", "a_ind"]])
    # Each owner keeps its share of both vintages: hh_1 three quarters.
    np.testing.assert_allclose(s.loc["hh_1", "f_cap_new"], 0.75 * 8.5)
    split.check_balance()
    assert "f_cap split into vintages" in split.provenance["note"].iloc[0]
    with pytest.raises(ValueError, match="between 0 and 1"):
        cge.split_vintages(sam, 1.5)
    with pytest.raises(ValueError, match="already"):
        cge.split_vintages(sam, 0.1, new="f_lab")
    with pytest.raises(ValueError, match="a_ind"):
        cge.split_vintages(sam, {"a_agr": 0.1})


CLOSURES = [cge.Closure(), cge.Closure(foreign="fixed_rate"),
            cge.Closure(investment="fixed_real"), cge.Closure(government="saving_rate")]


@pytest.mark.parametrize("closure", CLOSURES, ids=range(len(CLOSURES)))
def test_base_year_reproduces(closure):
    econ = two_sector()
    rep = cge.base_year_reproduction(econ, closure)
    assert rep["ok"], rep["checks"]
    hom = cge.homogeneity_test(econ, closure)
    assert hom["ok"], hom["checks"]
    res = cge.solve_period(econ, closure)
    np.testing.assert_allclose(res.vintage_value_added, [[58.5, 94.5], [6.5, 10.5]],
                               rtol=1e-12)


def test_one_elasticity_everywhere_is_the_model_without_vintages():
    """Old capital by activity and new capital, in one CES: another code path."""
    sigma = 0.7
    nested = two_sector(sigma_va=sigma, vintage_sigma=sigma, vintage_substitution=sigma)
    flat = cge.calibrate(cge.split_vintages(_frame(_BASE), 0.1),
                         toy.spec(factors=FACTORS, sigma_va=sigma,
                                  factor_mobility={"f_cap": 0.0}),
                         factor_pools={"f_cap": "activities"})
    a, b = cge.solve_period(nested, scenario=SHOCK), cge.solve_period(flat, scenario=SHOCK)
    for name in ("X", "Q", "pq", "hh_income", "factor_demand_by_activity"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-10, atol=1e-12,
                                   err_msg=name)
    np.testing.assert_allclose(a.factor_prices_by_activity, b.factor_prices_by_activity,
                               rtol=1e-10)
    assert abs(a.walras) < 1e-12


@pytest.mark.parametrize("scenario", [SHOCK, cge.Scenario(world_price=np.array([0.6, 1.0])),
                                      cge.Scenario(product_tax=np.array([0.2, 0.0]))])
def test_no_substitution_ex_post_holds_the_old_vintage_at_its_capital(scenario):
    econ = two_sector()
    res = cge.solve_period(econ, scenario=scenario)
    # Old capital over its share of value added (25 and 35 of 65 and 105).
    np.testing.assert_allclose(res.vintage_value_added[0],
                               [22.5 * 65 / 25, 31.5 * 105 / 35], rtol=1e-10)
    rent = res.factor_prices_by_activity[1]
    assert not np.allclose(rent, rent[0])          # each activity's old capital its own price


def test_the_one_good_steady_state_stays_put():
    """New capital is the year's investment, 20, which depreciation wears off 400."""
    path = cge.run_path(one_good(), years=YEARS)
    np.testing.assert_allclose(path.capital_stock, 400.0, rtol=1e-12)
    np.testing.assert_allclose(path.capital_by_pool["f_cap"]["a_x"], 380.0, rtol=1e-12)
    np.testing.assert_allclose(path.capital_by_pool["f_cap_new"]["f_cap_new"], 20.0, rtol=1e-12)
    gdp = [cge.real_gdp(p)["expenditure"] for p in path.periods]
    np.testing.assert_allclose(gdp, 100.0, rtol=1e-12)


def test_the_vintages_approach_one_capital_at_rate_one_over_the_elasticity():
    sam, _ = toy.build_one_good()
    plain = cge.calibrate(sam, cge.ModelSpec(factors=("f_lab", "f_cap"), sigma_va=0.8,
                                             **ONE_GOOD))
    growth = cge.Growth(labour=0.02, tfp=0.01)
    ref = cge.run_path(plain, years=YEARS, growth=growth)

    def gap(top):
        path = cge.run_path(one_good(top=top), years=YEARS, growth=growth)
        np.testing.assert_allclose(path.capital_stock, ref.capital_stock, rtol=0.05)
        return max(abs(cge.real_gdp(a)["expenditure"] / cge.real_gdp(b)["expenditure"] - 1)
                   for a, b in zip(path.periods, ref.periods))

    g80, g160, g320 = gap(80.0), gap(160.0), gap(320.0)
    assert g320 < g160 < g80 < 1e-4
    assert g80 / g160 == pytest.approx(2.0, rel=0.05)
    assert g160 / g320 == pytest.approx(2.0, rel=0.05)


def test_putty_clay_moves_new_capital_and_leaves_the_old():
    """Industry's world price rises: only new capital can follow it at first."""
    shock = cge.Scenario(world_price=np.array([1.0, 1.15]))
    clay = cge.run_path(two_sector(), years=YEARS, scenario=shock)
    mobile = cge.run_path(toy.build()[1], years=YEARS, scenario=shock)
    base = toy.build()[1].X0
    up_clay = clay.periods[0].X[1] / base[1] - 1
    up_mobile = mobile.periods[0].X[1] / base[1] - 1
    assert 0 < up_clay < up_mobile
    # Industry installs more than its share of new capital each year, so its
    # old capital grows, and its response catches up with the mobile economy's.
    old = clay.capital_by_pool["f_cap"]
    share = old["a_ind"] / old.sum(axis=1)
    assert share.iloc[-1] > share.iloc[1] > share.iloc[0]
    later = clay.periods[-1].X[1] / mobile.periods[-1].X[1]
    first = clay.periods[0].X[1] / mobile.periods[0].X[1]
    assert first < later
    assert clay.converged and clay.worst_walras < 1e-9


def test_what_vintages_refuse():
    sam = cge.split_vintages(_frame(_BASE), 0.1)
    spec = dict(factors=FACTORS, vintages={"f_cap": "f_cap_new"})
    with pytest.raises(ValueError, match="mobile"):
        toy.spec(factor_mobility={"f_cap_new": 1.0}, **spec)
    with pytest.raises(ValueError, match="numeraire"):
        toy.spec(numeraire_factor="f_cap_new", **spec)
    with pytest.raises(ValueError, match="one pair"):
        toy.spec(factors=FACTORS, vintages={"f_cap": "f_cap_new", "f_lab": "f_cap"})
    with pytest.raises(ValueError, match="energy"):
        toy.spec(energy=["c_ind"], **spec)
    with pytest.raises(ValueError, match="value_added"):
        toy.spec(value_added="cobb_douglas", **spec)
    with pytest.raises(ValueError, match="factor_pools"):
        cge.calibrate(sam, toy.spec(**spec), factor_pools={"f_cap": "activities"})
    one = cge.split_vintages(_frame(_BASE), {"a_agr": 0.0, "a_ind": 0.1})
    with pytest.raises(ValueError, match="a_agr"):
        cge.calibrate(one, toy.spec(**spec))
    econ = two_sector()
    with pytest.raises(ValueError, match="old vintage"):
        cge.run_path(econ, years=YEARS[:2],
                     growth=cge.Growth(capital_factor=("f_cap", "f_cap_new")))
    with pytest.raises(ValueError, match="FDI"):
        cge.run_path(econ, years=YEARS[:2], scenario=cge.Scenario(fdi={"f_cap": 1.0}))


def test_the_old_vintage_keeps_the_technique_it_was_installed_with():
    """Capital grows and labour does not, so each year's new capital is installed
    with less labour, and the old vintage inherits it."""
    path = cge.run_path(two_sector(), years=YEARS)
    ratio = [p.econ.vintage.theta_old[1] / p.econ.vintage.theta_old[0] for p in path.periods]
    ratio = np.array(ratio)                         # old capital per unit of labour
    np.testing.assert_allclose(ratio[0], [25 / 40, 35 / 70], rtol=1e-12)
    np.testing.assert_allclose(ratio[1], ratio[0], rtol=1e-12)    # the base year installs
    assert (np.diff(ratio[1:], axis=0) > 0).all()            # the base technique
    assert path.converged and path.worst_walras < 1e-9
    # With one elasticity in both bundles, and both capitals at one price, the
    # technique installed is the one the old vintage already has.
    v = two_sector(sigma_va=0.6, vintage_sigma=0.6).vintage
    w = np.array([[1.3, 1.3], [0.8, 0.8], [0.8, 0.8]])
    after = v.installed(w, np.array([[50.0, 90.0], [5.0, 12.0]]))
    np.testing.assert_allclose(after.theta_old, v.theta_old, rtol=1e-12)
