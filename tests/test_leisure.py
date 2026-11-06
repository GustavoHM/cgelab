"""The labour–leisure choice: households split their time, and the labour market clears on it."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy

SAM = toy._frame(toy._BASE)
RATIO = {"hh_1": 0.5, "hh_2": 0.8}
ECON = cge.calibrate(SAM, toy.spec(numeraire_factor="f_cap", leisure_ratio=RATIO))


def test_the_base_year_reproduces_with_the_calibrated_leisure():
    period = cge.solve_period(ECON)
    assert period.converged and abs(period.walras) < 1e-12
    labour = ECON.hh_facshare[:, 0] * ECON.fac_supply[0]
    np.testing.assert_allclose(period.leisure["labour"], labour, rtol=1e-12)
    np.testing.assert_allclose(period.leisure["leisure"], labour * [0.5, 0.8], rtol=1e-12)


@pytest.mark.parametrize("scenario", [
    cge.Scenario(productivity=np.array([1.1, 1.05])),
    cge.Scenario(factor_supply=np.array([110.0, 72.0])),
    cge.Scenario(world_price=np.array([1.3, 0.8])),
])
def test_each_household_is_on_its_cobb_douglas_and_the_market_clears(scenario):
    r = cge.solve_period(ECON, scenario=scenario)
    assert r.converged and abs(r.walras) < 1e-12
    L = r.leisure
    np.testing.assert_allclose(L["labour"] + L["leisure"], L["time"], rtol=1e-12)
    net = (1.0 - ECON.zeros_if_none("hh_dtax", 2)) * (1.0 - ECON.hh_savrate)
    # net w L = k C: the value of leisure over spending is the calibrated ratio.
    np.testing.assert_allclose(net * L["wage"] * L["leisure"], ECON.leisure["k"] * L["spending"],
                               rtol=1e-12)
    np.testing.assert_allclose(L["spending"], r.hh_commodity_spend, rtol=1e-12)
    assert r.factor_demand[0] == pytest.approx(L["labour"].sum(), rel=1e-10)


def test_a_higher_wage_from_more_capital_raises_labour_supply():
    base = cge.solve_period(ECON)
    more = cge.solve_period(ECON, scenario=cge.Scenario(factor_supply=np.array([110.0, 72.0])))
    assert more.factor_prices[0] > base.factor_prices[0]
    assert more.leisure["labour"].sum() != pytest.approx(base.leisure["labour"].sum(), rel=1e-6)


def test_a_vanishing_leisure_ratio_is_fixed_labour_supply():
    tiny = cge.calibrate(SAM, toy.spec(numeraire_factor="f_cap", leisure_ratio=1e-9))
    fixed = cge.calibrate(SAM, toy.spec(numeraire_factor="f_cap"))
    sc = cge.Scenario(productivity=np.array([1.1, 1.0]))
    a, b = cge.solve_period(tiny, scenario=sc), cge.solve_period(fixed, scenario=sc)
    np.testing.assert_allclose(a.factor_prices, b.factor_prices, rtol=1e-7)


def test_leisure_is_refused_where_it_would_contradict_something():
    with pytest.raises(ValueError, match="numeraire"):
        cge.calibrate(SAM, toy.spec(leisure_ratio=0.5))          # f_lab is the numeraire
    with pytest.raises(ValueError, match="full_employment"):
        cge.solve_period(ECON, cge.Closure(labour="fixed_real_wage"))
    with pytest.raises(ValueError, match="wedges"):
        cge.solve_period(ECON, scenario=cge.Scenario(
            consumer_wedge=cge.consumer_wedge(ECON, {"c_ind": 0.1})))
    with pytest.raises(KeyError, match="not households"):
        cge.calibrate(SAM, toy.spec(numeraire_factor="f_cap", leisure_ratio={"hh_9": 1.0}))
