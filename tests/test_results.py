"""Employment, real GDP and the sectoral panel. Hand arithmetic, not the solver's memory."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy


def test_factor_demand_by_activity_sums_to_the_solved_demand():
    _, econ = toy.build()
    period = cge.solve_period(econ)
    by_act = cge.factor_demand_by_activity(period)
    assert list(by_act.index) == list(econ.factors)
    assert list(by_act.columns) == list(econ.acts)
    # The toy's factor block, written out: agriculture 40 and 25, industry 70 and 35.
    assert by_act.loc["f_lab", "a_agr"] == pytest.approx(40.0)
    assert by_act.loc["f_cap", "a_agr"] == pytest.approx(25.0)
    assert by_act.loc["f_lab", "a_ind"] == pytest.approx(70.0)
    assert by_act.loc["f_cap", "a_ind"] == pytest.approx(35.0)
    assert np.allclose(by_act.sum(axis=1), period.factor_demand)


def test_real_gdp_at_the_base_year_is_the_sam():
    _, econ = toy.build()
    gdp = cge.real_gdp(cge.solve_period(econ))
    # C 135 + G 20 + I 50 + E 30 - M 40 = 195. Factors 110 + 60 = 170.
    # Product tax 15, activity tax 10. Nothing else.
    assert gdp["expenditure"] == pytest.approx(195.0)
    assert gdp["value_added"] == pytest.approx(170.0)
    assert gdp["product_tax"] == pytest.approx(15.0)
    assert gdp["activity_tax"] == pytest.approx(10.0)
    assert gdp["own_account"] == pytest.approx(0.0)
    assert gdp["gap"] == pytest.approx(25.0)
    assert gdp["other"] == pytest.approx(0.0, abs=1e-9)


def test_margins_do_not_open_a_gap():
    """A margin is purchased and produced. It is on both sides, so the gap stays the tax."""
    _, econ = toy.build_with_margins()
    gdp = cge.real_gdp(cge.solve_period(econ))
    assert gdp["expenditure"] == pytest.approx(207.0)
    assert gdp["value_added"] == pytest.approx(182.0)
    assert gdp["other"] == pytest.approx(0.0, abs=1e-9)


def test_own_account_production_is_in_expenditure():
    """Leaving it out makes value added larger by 9 and the gap looks like a leak."""
    _, econ = toy.build_with_own_production()
    gdp = cge.real_gdp(cge.solve_period(econ))
    assert gdp["own_account"] == pytest.approx(9.0)
    assert gdp["value_added"] == pytest.approx(179.0)
    assert gdp["gap"] == pytest.approx(gdp["product_tax"] + gdp["activity_tax"])
    assert gdp["other"] == pytest.approx(0.0, abs=1e-9)


def test_a_tax_shock_moves_the_gap_by_the_tax():
    _, econ = toy.build()
    shocked = cge.solve_period(econ, scenario=cge.Scenario(product_tax=econ.ptax * 1.2))
    gdp = cge.real_gdp(shocked)
    assert abs(shocked.walras) < 1e-9
    named = gdp["product_tax"] + gdp["activity_tax"] + gdp["other"]
    assert gdp["gap"] == pytest.approx(named)
    # The residual past the tax is the base-price identity leaving the calibrated point.
    assert abs(gdp["other"]) < 1e-3
    assert abs(gdp["gap"] - named) < 1e-9


def test_sectoral_path_covers_every_account_and_year():
    _, econ = toy.build()
    path = cge.run_path(econ, years=[2021, 2022], growth=cge.Growth())
    frame = cge.sectoral_path(path)
    assert path.sectoral().equals(frame)
    assert set(frame["year"]) == {2021, 2022}
    commodities = frame[frame["kind"] == "commodity"]
    activities = frame[frame["kind"] == "activity"]
    assert set(commodities["account"]) == set(econ.coms)
    assert set(activities["account"]) == set(econ.acts)
    agr = commodities[(commodities["year"] == 2021) & (commodities["account"] == "c_agr")].iloc[0]
    assert agr["Q"] == pytest.approx(path[2021].Q[econ.commodity("c_agr")])
    assert agr["pq"] == pytest.approx(path[2021].pq[econ.commodity("c_agr")])
    assert np.isnan(agr["X"])
    ind = activities[(activities["year"] == 2021) & (activities["account"] == "a_ind")].iloc[0]
    assert ind["X"] == pytest.approx(path[2021].X[econ.activity("a_ind")])
    assert np.isnan(ind["Q"])
