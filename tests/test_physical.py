"""Physical units alongside the money."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy

UNITS = dict(commodities={"c_ind": ("GWh", 1000.0)}, activities={"a_agr": ("t", 500.0)})


@pytest.mark.parametrize("build", [toy.build, toy.build_with_margins,
                                   toy.build_with_trade_taxes], ids=lambda b: b.__name__)
def test_the_base_year_returns_the_totals_and_supply_meets_use(build):
    econ = cge.add_physical_units(build()[1], **UNITS)
    p = cge.physical(cge.solve_period(econ))
    assert p["supply"]["c_ind"] == ("GWh", pytest.approx(1000.0, rel=1e-12))
    assert p["output"]["a_agr"] == ("t", pytest.approx(500.0, rel=1e-12))
    assert p["use"]["c_ind"].sum() == pytest.approx(1000.0, rel=1e-12)


def test_supply_meets_use_away_from_the_base_year():
    econ = cge.add_physical_units(toy.build_with_margins()[1], **UNITS)
    p = cge.physical(cge.solve_period(econ, scenario=cge.Scenario(product_tax=econ.ptax * 1.5)))
    assert p["use"]["c_ind"].sum() == pytest.approx(p["supply"]["c_ind"][1], rel=1e-12)


def test_a_physical_coefficient_becomes_the_sam_entry():
    econ = cge.add_physical_units(toy.build_with_margins()[1], **UNITS)
    c, j = econ.commodity("c_ind"), econ.activity("a_agr")
    p = cge.physical(cge.solve_period(econ))
    unit, amount = p["intensity"][("c_ind", "a_agr")]
    assert unit == "GWh/t"
    entry = cge.physical_coefficient(
        amount, input_per_value=econ.physical_units.commodities["c_ind"][1] * econ.arm_share[c],
        output_per_value=econ.physical_units.activities["a_agr"][1], input_price=1 + econ.ptax[c])
    assert entry == pytest.approx(10.0 / 100.0, rel=1e-12)    # the SAM's c_ind -> a_agr over output


def test_intensity_is_fixed_under_leontief_and_moves_under_an_energy_nest():
    tax = cge.Scenario(product_tax=None, carbon_price=1.0)
    fuel = pd.DataFrame({"a_agr": [0.0, 3.0], "a_ind": [0.0, 12.0]}, index=["c_agr", "c_ind"])
    leo = cge.add_emissions(cge.add_physical_units(toy.build()[1], **UNITS), intermediate=fuel)
    nest = cge.add_emissions(cge.add_physical_units(toy.build(model_spec=toy.spec(
        energy=["c_ind"], sigma_vae=0.6))[1], **UNITS), intermediate=fuel)
    for econ, moves in ((leo, False), (nest, True)):
        base = cge.physical(cge.solve_period(econ))["intensity"][("c_ind", "a_agr")][1]
        shock = cge.physical(cge.solve_period(econ, scenario=tax))["intensity"][("c_ind", "a_agr")][1]
        assert (abs(shock / base - 1) > 1e-6) == moves


def test_misuse_is_refused():
    _, econ = toy.build()
    with pytest.raises(ValueError, match="no physical units"):
        cge.physical(cge.solve_period(econ))
    with pytest.raises(ValueError, match="positive"):
        cge.physical_coefficient(56.0, input_per_value=0.0, output_per_value=1.0)
