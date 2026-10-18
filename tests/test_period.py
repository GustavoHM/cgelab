"""Within-period equilibrium: base year, homogeneity, Walras, closures, wedges."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy

VARIANTS = [("plain", toy.build), ("margins", toy.build_with_margins),
            ("own_account", toy.build_with_own_production)]


@pytest.mark.parametrize("label,build", VARIANTS)
def test_base_year_reproduces_the_sam(label, build):
    """With no shock every price is one and every quantity its base-year value."""
    _, econ = build()
    rep = cge.base_year_reproduction(econ)
    assert rep["ok"], f"{label}: worst {rep['worst']:.2e}, checks {rep['checks']}"
    assert abs(rep["walras"]) < 1e-10


@pytest.mark.parametrize("label,build", VARIANTS)
def test_prices_are_homogeneous_of_degree_zero(label, build):
    """Scaling the numeraire must leave every real magnitude untouched."""
    _, econ = build()
    rep = cge.homogeneity_test(econ, factor=3.0)
    assert rep["ok"], f"{label}: worst {rep['worst']:.2e}, checks {rep['checks']}"


CLOSURES = [
    cge.Closure(),
    cge.Closure(foreign="fixed_rate"),
    cge.Closure(investment="fixed_real"),
    cge.Closure(government="nominal_fixed"),
    cge.Closure(investment="fixed_real", foreign="fixed_rate"),
]


@pytest.mark.parametrize("closure", CLOSURES, ids=lambda c: c.describe()[12:])
def test_walras_holds_under_a_shock_in_every_closure(closure):
    """The market the solver was NOT told to clear must clear anyway.

    A closure that pins a magnitude without freeing an adjusting variable
    somewhere else does not close the model, and the symptom is exactly this
    residual rather than a solver failure.
    """
    _, econ = toy.build()
    res = cge.solve_period(econ, closure, cge.Scenario(name="tax", product_tax=econ.ptax * 1.2))
    assert res.converged
    assert abs(res.walras) < 1e-9


def test_wedge_revenue_recycled_equals_wedge_revenue_paid():
    """An infinite series, not a single pass: the revenue raises income, which raises it again."""
    _, econ = toy.build()
    W = cge.consumer_wedge(econ, {"c_ind": 0.08})
    res = cge.solve_period(econ, scenario=cge.Scenario(name="wedge", consumer_wedge=W))
    assert res.hh_wedge_paid.sum() == pytest.approx(res.wedge_revenue, rel=1e-12)
    assert abs(res.walras) < 1e-9


def test_the_recycling_rule_changes_the_incidence():
    """Whom the revenue goes to is a modelling choice with a distributional sign."""
    _, econ = toy.build()
    W = cge.consumer_wedge(econ, {"c_ind": 0.08})
    cap = cge.solve_period(econ, scenario=cge.Scenario(consumer_wedge=W, wedge_revenue_to="capital"))
    lump = cge.solve_period(econ, scenario=cge.Scenario(consumer_wedge=W, wedge_revenue_to="lumpsum"))
    half = cge.solve_period(econ, scenario=cge.Scenario(consumer_wedge=W, wedge_revenue_to=0.5))
    # hh_1 owns three quarters of capital, so it gains under "capital" and loses
    # under "lumpsum"; the convex combination lies between the two.
    assert cap.hh_income[0] > lump.hh_income[0]
    assert lump.hh_income[0] < half.hh_income[0] < cap.hh_income[0]


def test_an_invalid_recycling_weight_is_refused():
    _, econ = toy.build()
    W = cge.consumer_wedge(econ, {"c_ind": 0.05})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cge.solve_period(econ, scenario=cge.Scenario(consumer_wedge=W, wedge_revenue_to=1.4))


def test_government_wedge_costs_the_budget_and_is_not_recycled():
    """A refund at the till lowers public saving; it does not return to anyone's income."""
    _, econ = toy.build()
    Wg = -cge.consumer_wedge(econ, {"c_agr": 0.05})
    res = cge.solve_period(econ, scenario=cge.Scenario(name="cashback", government_wedge=Wg))
    assert res.government_cost > 0.0
    assert res.wedge_revenue == 0.0
    assert abs(res.walras) < 1e-9


def test_product_tax_raises_the_consumer_price_and_lowers_real_income():
    _, econ = toy.build()
    base = cge.solve_period(econ)
    shock = cge.solve_period(econ, scenario=cge.Scenario(product_tax=econ.ptax * 1.5))
    assert np.all(shock.pq > base.pq)
    assert np.all(shock.real["hh_real_income"] < base.real["hh_real_income"])


def test_les_damps_the_quantity_response_relative_to_cobb_douglas():
    """Subsistence quantities do not respond to price; only the supernumerary part does."""
    _, cd = toy.build()
    _, les = toy.build(model_spec=toy.spec(household_demand="les", frisch=-2.0))
    j = cd.commodity("c_ind")
    shock = cge.Scenario(product_tax=np.array([cd.ptax[0], cd.ptax[1] * 2.0]))
    cd_move = abs(cge.solve_period(cd, scenario=shock).hh_quantity[:, j].sum()
                  / cge.solve_period(cd).hh_quantity[:, j].sum() - 1.0)
    les_move = abs(cge.solve_period(les, scenario=shock).hh_quantity[:, j].sum()
                   / cge.solve_period(les).hh_quantity[:, j].sum() - 1.0)
    assert les_move < cd_move


def test_scaling_a_market_activity_is_refused():
    """Its level is already determined by the commodity balance."""
    _, econ = toy.build_with_own_production()
    with pytest.raises(ValueError, match="market share"):
        cge.solve_period(econ, scenario=cge.Scenario(activity_scale={"a_ind": 2.0}))


def test_own_account_activity_scales_and_keeps_walras():
    _, econ = toy.build_with_own_production()
    j = econ.activity("a_own")
    res = cge.solve_period(econ, scenario=cge.Scenario(activity_scale={"a_own": 2.0}))
    assert res.X[j] == pytest.approx(2.0 * econ.X0[j], rel=1e-12)
    assert abs(res.walras) < 1e-9


def test_factor_supply_shock_moves_the_relative_price_the_right_way():
    _, econ = toy.build()
    base = cge.solve_period(econ)
    more_capital = econ.fac_supply.copy()
    more_capital[econ.factor("f_cap")] *= 1.2
    res = cge.solve_period(econ, scenario=cge.Scenario(factor_supply=more_capital))
    assert res.factor_prices[econ.factor("f_cap")] < base.factor_prices[econ.factor("f_cap")]


def _own_account_with_activity_tax():
    """`a_own` pays 1 of activity tax; agriculture pays 1 less and hires 1 more capital."""
    import pandas as pd

    sam, _ = toy.build_with_own_production()
    f = sam.frame.set_index(["row", "col"])["value"].to_dict()
    f.update({("f_cap", "a_own"): 8.0, ("gov", "a_own"): 1.0,
              ("gov", "a_agr"): 4.0, ("f_cap", "a_agr"): 26.0})
    out = cge.SAM.from_long(pd.DataFrame([(r, c, v) for (r, c), v in f.items()],
                                         columns=["row", "col", "value"]))
    out.check_balance()
    return cge.calibrate(out, toy.spec())


def test_an_own_account_activity_tax_reaches_government():
    """Households pay it in the price of what they buy straight from the activity.
    Counting it only on market activities left it unreceived, and Walras off by
    the tax."""
    econ = _own_account_with_activity_tax()
    assert econ.atax[econ.activity("a_own")] > 0
    rep = cge.base_year_reproduction(econ)
    assert rep["ok"], rep["checks"]
    scaled = cge.solve_period(econ, scenario=cge.Scenario(activity_scale={"a_own": 2.0}))
    assert abs(scaled.walras) < 1e-9
    assert scaled.government["activity_tax"] == pytest.approx(
        float(econ.atax @ scaled.X), rel=1e-12)
