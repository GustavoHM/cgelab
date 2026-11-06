"""The generic closure swap: swapping and swapping back reproduce the standard closure."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy

_, ECON = toy.build(toy.spec(export_rule="cet"))
_, TAXED = toy.build_with_export_tax()


def _same(a, b, rtol=1e-10):
    for field in ("factor_prices", "X", "Q", "pq", "imports", "exports", "hh_income"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=rtol, atol=1e-10)
    assert a.exchange_rate == pytest.approx(b.exchange_rate, rel=rtol)


@pytest.mark.parametrize("given, swaps", [
    (dict(productivity=np.array([1.05, 1.0])), {"X[a_agr]": "productivity[a_agr]"}),
    (dict(world_price=np.array([1.0, 1.1])), {"imports[c_ind]": "world_price[c_ind]"}),
    (dict(export_price=np.array([1.2, 1.0])), {"exports[c_agr]": "export_price[c_agr]"}),
    (dict(factor_supply=np.array([115.0, 60.0])), {"factor_prices[f_cap]": "factor_supply[f_lab]"}),
    (dict(productivity=np.array([1.04, 0.98]), world_price=np.array([1.0, 1.15])),
     {"X[a_agr]": "productivity[a_agr]", "X[a_ind]": "productivity[a_ind]",
      "imports[c_ind]": "world_price[c_ind]"}),
])
def test_swapping_recovers_the_instrument_and_the_standard_solution(given, swaps):
    scenario = cge.Scenario(**given)
    standard = cge.solve_period(ECON, scenario=scenario)
    targets = {}
    for variable in swaps:
        attr, label = variable.rstrip("]").split("[")
        names = {"X": ECON.acts, "imports": ECON.coms, "exports": ECON.coms, "pq": ECON.coms,
                 "factor_prices": ECON.factors}[attr]
        targets[variable] = float(getattr(standard, attr)[names.index(label)])
    swapped = cge.solve_swapped(ECON, exogenous=targets, endogenous=list(swaps.values()))
    _same(swapped.period, standard)
    # Swapping back: the scenario it found, under the standard closure, is the same period.
    _same(cge.solve_period(ECON, scenario=swapped.scenario), standard)
    for instrument, value in swapped.instruments.items():
        field, _, label = instrument.rstrip("]").partition("[")
        own = getattr(scenario, field)
        if label:
            axis = {"productivity": ECON.acts, "world_price": ECON.coms,
                    "export_price": ECON.coms, "factor_supply": ECON.factors}[field]
            own = own[axis.index(label)]
        assert value == pytest.approx(float(own), rel=1e-9)


def test_investment_under_a_fixed_real_closure():
    closure = cge.Closure(investment="fixed_real")
    standard = cge.solve_period(ECON, closure, cge.Scenario(investment_level=55.0))
    swapped = cge.solve_swapped(ECON, closure, exogenous={"X[a_ind]": standard.X[1]},
                                endogenous=["investment_level"])
    assert swapped.instruments["investment_level"] == pytest.approx(55.0, rel=1e-9)
    # Under savings-driven investment the level is not read, and the swap is refused.
    with pytest.raises(ValueError, match="not identified"):
        cge.solve_swapped(ECON, exogenous={"X[a_ind]": standard.X[1]},
                          endogenous=["investment_level"])


def test_a_rate_is_solved_in_levels():
    rates = TAXED.te0.copy()
    rates[1] = 0.12
    standard = cge.solve_period(TAXED, scenario=cge.Scenario(export_tax=rates))
    swapped = cge.solve_swapped(TAXED, exogenous={"exports[c_ind]": standard.exports[1]},
                                endogenous=["export_tax[c_ind]"])
    assert swapped.instruments["export_tax[c_ind]"] == pytest.approx(0.12, rel=1e-9)
    # Abolishing to hit a target: the rate can cross zero.
    up = cge.solve_period(TAXED, scenario=cge.Scenario(export_tax=TAXED.te0 * [1, -1]))
    neg = cge.solve_swapped(TAXED, exogenous={"exports[c_ind]": up.exports[1]},
                            endogenous=["export_tax[c_ind]"])
    assert neg.instruments.iloc[0] == pytest.approx(-TAXED.te0[1], rel=1e-8)


def test_the_historical_closure_is_a_case_of_it():
    """calibrate_historical's common productivity is productivity[*] against real GDP."""
    _, econ = toy.build()
    observed = cge.solve_period(econ, scenario=cge.Scenario(
        productivity=np.array([1.07, 1.02])))
    gdp = cge.real_gdp(observed)["expenditure"]
    hist = cge.calibrate_historical(
        econ, years=[2021], targets={"gdp": (pd.Series({2021: gdp}),
                                             lambda r: cge.real_gdp(r)["expenditure"])},
        shifters={"gdp": ("productivity",)})
    swapped = cge.solve_swapped(econ, exogenous={"real_gdp": gdp},
                                endogenous=["productivity[*]"])
    np.testing.assert_allclose(hist.multiplier.loc[2021].to_numpy(),
                               swapped.instruments["productivity[*]"], rtol=1e-9)
    _same(swapped.period, hist.path[2021])


def test_a_callable_and_a_government_target():
    standard = cge.solve_period(ECON, scenario=cge.Scenario(product_tax=ECON.ptax * 1.3))
    swapped = cge.solve_swapped(
        ECON, exogenous={"government.saving": standard.government["saving"]},
        endogenous=["product_tax[*]"])
    np.testing.assert_allclose(swapped.scenario.product_tax, ECON.ptax * 1.3, rtol=1e-8)
    by_hand = cge.solve_swapped(ECON, exogenous={lambda p: p.hh_income[0]: standard.hh_income[0]},
                                endogenous=["product_tax[*]"])
    assert by_hand.period.hh_income[0] == pytest.approx(standard.hh_income[0], rel=1e-10)
    assert list(by_hand.achieved.index) == ["variable 0"]


def test_unbalanced_or_unknown_swaps_are_refused():
    with pytest.raises(ValueError, match="one for one"):
        cge.solve_swapped(ECON, exogenous={"X[a_agr]": 100.0}, endogenous=[])
    with pytest.raises(KeyError, match="not one of"):
        cge.solve_swapped(ECON, exogenous={"X[a_agr]": 100.0}, endogenous=["sigma_va"])
    with pytest.raises(KeyError, match="accounts"):
        cge.solve_swapped(ECON, exogenous={"X[a_agr]": 100.0},
                          endogenous=["productivity[c_agr]"])
    with pytest.raises(KeyError, match="variable"):
        cge.solve_swapped(ECON, exogenous={"Z[a_agr]": 100.0},
                          endogenous=["productivity[a_agr]"])
    with pytest.raises(ValueError, match="not calibrated"):
        cge.solve_swapped(ECON, exogenous={"X[a_agr]": 100.0}, endogenous=["tariff[c_agr]"])
    with pytest.raises(RuntimeError, match="did not reach"):
        cge.solve_swapped(ECON, exogenous={"X[a_agr]": -5.0},
                          endogenous=["productivity[a_agr]"])
