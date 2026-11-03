"""What a fixed_real closure can finance, and the error that says why not."""

import pytest

import cgelab as cge
from cgelab.examples import toy


def test_bounds_bracket_the_levels_that_solve():
    _, econ = toy.build_with_institutions()
    closure = cge.Closure(investment="fixed_real")
    b = cge.investment_bounds(econ, closure)
    assert b.reference == pytest.approx(50.0)
    assert b.low == 0.0 and b.below is None
    assert 100.0 < b.high < 200.0
    assert cge.solve_period(econ, closure, cge.Scenario(investment_level=b.high)).converged
    assert "hh_2" in b.above and "largest levy" in b.above


def test_a_level_beyond_the_bound_names_the_household_and_the_levy():
    _, econ = toy.build_with_institutions()
    closure = cge.Closure(investment="fixed_real")
    with pytest.raises(RuntimeError, match="every start point failed") as info:
        cge.solve_period(econ, closure, cge.Scenario(investment_level=300.0))
    text = str(info.value)
    assert "hh_2" in text and "investment_level=300" in text and "largest levy" in text
    assert isinstance(info.value.__cause__, ValueError)


def test_a_reference_that_does_not_solve_is_refused():
    _, econ = toy.build_with_institutions()
    with pytest.raises(ValueError, match="no bracket"):
        cge.investment_bounds(econ, scenario=cge.Scenario(investment_level=300.0))
