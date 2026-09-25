"""Recursive dynamics: accumulation, growth and the accounting along the path."""

import numpy as np
import pytest

import cgekit as ck
from cgekit.examples import toy

GROWTH = [ck.Growth(), ck.Growth(labour=0.01), ck.Growth(tfp=0.015),
          ck.Growth(labour=0.01, tfp=0.015)]


@pytest.mark.parametrize("growth", GROWTH,
                         ids=lambda g: f"lab{g.labour}-tfp{g.tfp}")
def test_walras_holds_at_every_date(growth):
    """A residual that GROWS along the path is the signature of a coefficient that
    the price block saw and the quantity block did not."""
    _, econ = toy.build()
    path = ck.run_path(econ, years=range(2021, 2041), growth=growth)
    assert path.converged
    assert path.worst_walras < 1e-9


def test_a_static_path_stays_at_the_base_year():
    """No growth and no depreciation gap: the economy must not drift on its own."""
    _, econ = toy.build()
    # Depreciation exactly absorbing base investment keeps the stock constant.
    _, e2 = toy.build()
    base = ck.solve_period(e2)
    delta = float(base.real["inv_real"]) / (e2.fac_supply[e2.factor("f_cap")] / 0.10)
    path = ck.run_path(e2, years=range(2021, 2031),
                       growth=ck.Growth(depreciation=delta, discount=0.10 - delta))
    agg = path.aggregates()
    assert np.allclose(agg["price_f_cap"], 1.0, atol=1e-8)
    assert np.allclose(agg["hh_real"], agg["hh_real"].iloc[0], rtol=1e-8)


def test_capital_accumulates_from_real_investment_not_from_the_service_flow():
    """Adding investment to the SAM's capital ROW compounds a flow onto a flow."""
    _, econ = toy.build()
    growth = ck.Growth(depreciation=0.05, discount=0.05)
    path = ck.run_path(econ, years=range(2021, 2031), growth=growth)
    k = path.capital_stock
    assert k[0] == pytest.approx(econ.fac_supply[econ.factor("f_cap")] / growth.user_cost)
    assert np.all(np.diff(k) > 0)
    # Twenty-five years of this cannot multiply the stock by an order of magnitude.
    assert k[-1] / k[0] < 3.0


def test_a_per_year_scenario_follows_a_calendar():
    _, econ = toy.build()
    W = ck.consumer_wedge(econ, {"c_ind": 0.05})

    def calendar(year, t):
        return ck.Scenario(name="phase-in", consumer_wedge=W if year >= 2025 else None)

    path = ck.run_path(econ, years=range(2021, 2031), scenario=calendar)
    paid = [float(p.hh_wedge_paid.sum()) for p in path.periods]
    assert all(v == 0.0 for v in paid[:4])
    assert all(v > 0.0 for v in paid[4:])


def test_scenario_factor_supply_overrides_the_growth_block():
    _, econ = toy.build()
    fixed = econ.fac_supply.copy()
    path = ck.run_path(econ, years=range(2021, 2026),
                       growth=ck.Growth(labour=0.05),
                       scenario=lambda y, t: ck.Scenario(factor_supply=fixed))
    lab = econ.factor("f_lab")
    dem = [float(p.factor_demand[lab]) for p in path.periods]
    assert np.allclose(dem, fixed[lab], rtol=1e-9)
