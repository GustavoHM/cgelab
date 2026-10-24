"""Sector-specific capital in a recursive path: a stock per activity.

The check that matters comes from another code path. A capital factor with a
pool per activity, at ω = 0, is a set of sector-specific factors, and the path
already accumulates several capital factors separately. Both must give the same
path, year by year, whatever the rule that shares investment out.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.examples.toy import _BASE, _frame

YEARS = list(range(2021, 2036))


def pooled(omega=0.0):
    sp = toy.spec(factor_mobility={"f_cap": omega})
    return cge.calibrate(_frame(_BASE), sp, factor_pools={"f_cap": "activities"})


def split_by_hand():
    """The toy with capital written out as one factor per activity.

    Each household owns the same share of both, its share of capital income in
    the toy (45/60 and 15/60).
    """
    drop = {("f_cap", "a_agr", 25.0), ("f_cap", "a_ind", 35.0),
            ("hh_1", "f_cap", 45.0), ("hh_2", "f_cap", 15.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [("f_ka", "a_agr", 25.0), ("f_ki", "a_ind", 35.0),
                ("hh_1", "f_ka", 25 * 0.75), ("hh_2", "f_ka", 25 * 0.25),
                ("hh_1", "f_ki", 35 * 0.75), ("hh_2", "f_ki", 35 * 0.25)]
    return cge.calibrate(_frame(entries), toy.spec(factors=("f_lab", "f_ka", "f_ki")))


SHOCK = cge.Scenario(world_price=np.array([1.0, 1.15]))


@pytest.mark.parametrize("allocation", ["proportional", 2.0])
def test_a_pool_per_activity_is_capital_split_by_hand(allocation):
    growth = dict(labour=0.01, tfp=0.01, allocation=allocation)
    a = cge.run_path(pooled(), years=YEARS, scenario=SHOCK,
                     growth=cge.Growth(**growth))
    b = cge.run_path(split_by_hand(), years=YEARS, scenario=SHOCK,
                     growth=cge.Growth(capital_factor=("f_ka", "f_ki"), **growth))
    stocks = a.capital_by_pool["f_cap"]
    assert list(stocks.columns) == ["a_agr", "a_ind"]
    np.testing.assert_allclose(stocks.to_numpy(), b.capital_stock, rtol=1e-10)
    np.testing.assert_allclose(a.capital_stock, b.capital_stock.sum(axis=1), rtol=1e-10)
    for p, q in zip(a.periods, b.periods):
        np.testing.assert_allclose(p.X, q.X, rtol=1e-10)
        np.testing.assert_allclose(p.factor_prices_by_activity[1], q.factor_prices[1:],
                                   rtol=1e-10)
    assert a.worst_walras < 1e-9


def test_proportional_allocation_keeps_the_composition_and_rents_move_it():
    flat = cge.run_path(pooled(), years=YEARS, growth=cge.Growth(labour=0.01))
    k = flat.capital_by_pool["f_cap"]
    np.testing.assert_allclose(k["a_agr"] / k.sum(axis=1), 25 / 60, rtol=1e-12)
    # Industry's price rises with its world price, so its capital earns more and
    # draws a growing share of investment when investment follows rents.
    led = cge.run_path(pooled(), years=YEARS, scenario=SHOCK,
                       growth=cge.Growth(labour=0.01, allocation=3.0))
    share = led.capital_by_pool["f_cap"]["a_ind"] / led.capital_stock
    assert share.iloc[-1] > share.iloc[0] > 35 / 60 - 1e-12
    rent = led.periods[0].factor_prices_by_activity[1]
    assert rent[1] > rent[0]


def test_pools_that_are_not_activities_accumulate_too():
    """Two owners of capital, each mostly in one sector: a stock per owner."""
    pools = pd.DataFrame([[20.0, 5.0], [5.0, 30.0]], index=["north", "south"],
                         columns=["a_agr", "a_ind"])
    econ = cge.calibrate(_frame(_BASE), toy.spec(factor_mobility={"f_cap": 1.5}),
                         factor_pools={"f_cap": pools})
    path = cge.run_path(econ, years=YEARS, growth=cge.Growth(labour=0.01))
    k = path.capital_by_pool["f_cap"]
    assert list(k.columns) == ["north", "south"]
    np.testing.assert_allclose(k.iloc[0], [250.0, 350.0], rtol=1e-12)   # services / 0.10
    np.testing.assert_allclose(k["north"] / k["south"], 25 / 35, rtol=1e-12)
    assert path.converged and path.worst_walras < 1e-9


def test_fdi_into_sectoral_capital_is_shared_and_foreign_owned():
    econ = pooled()
    path = cge.run_path(econ, years=YEARS[:4], growth=cge.Growth(allocation=2.0),
                        scenario=cge.Scenario(fdi={"f_cap": 3.0}))
    assert path.foreign_capital[0] == 0.0 and path.foreign_capital[1] > 0.0
    assert path.periods[1].scenario.foreign_factor_share[1] > 0.0
    np.testing.assert_allclose(path.capital_by_pool["f_cap"].sum(axis=1), path.capital_stock,
                               rtol=1e-12)


def test_a_scenario_that_sets_the_pools_overrides_the_stocks():
    econ = pooled()
    fixed = cge.Scenario(pool_supply={"f_cap": np.array([25.0, 35.0])})
    path = cge.run_path(econ, years=YEARS[:3], scenario=fixed,
                        growth=cge.Growth(labour=0.02))
    for p in path.periods:
        np.testing.assert_allclose(p.factor_pools["f_cap"]["endowment"], [25.0, 35.0])


def test_a_pool_per_activity_is_refused_for_a_mobile_factor():
    with pytest.raises(ValueError, match="factor_mobility"):
        cge.calibrate(_frame(_BASE), toy.spec(), factor_pools={"f_cap": "activities"})
