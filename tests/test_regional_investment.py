"""Fixed-real investment in the regional solver, with forced saving."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions


def test_fixed_real_reproduces_the_base_year():
    _, econ = regions.build()
    reference = cge.solve_period(econ)
    fixed = cge.solve_period(econ, cge.Closure(investment="fixed_real"))
    assert fixed.converged and abs(fixed.walras) < 1e-8
    assert fixed.real["inv_real"] == pytest.approx(40.0, rel=1e-10)
    np.testing.assert_allclose(fixed.X, reference.X, rtol=1e-10)
    np.testing.assert_allclose(fixed.Q, reference.Q, rtol=1e-10)
    assert fixed.savings == pytest.approx(40.0, rel=1e-10)


def test_an_investment_level_is_held_and_saving_closes():
    _, econ = regions.build()
    fixed = cge.solve_period(econ, cge.Closure(investment="fixed_real"),
                             cge.Scenario(investment_level=80.0))
    assert fixed.converged and abs(fixed.walras) < 1e-8
    assert fixed.real["inv_real"] == pytest.approx(80.0, rel=1e-10)
    assert fixed.savings == pytest.approx(80.0, rel=1e-10)
    assert fixed.investment_quantity.sum() == pytest.approx(80.0, rel=1e-10)
    assert fixed.household_saving + fixed.forced_saving == pytest.approx(80.0, rel=1e-10)
    assert fixed.forced_saving > 0.0
    assert fixed.foreign_saving == 0.0


def test_invalid_investment_levels_are_refused():
    _, econ = regions.build()
    with pytest.raises(ValueError, match="nonnegative"):
        cge.solve_period(econ, cge.Closure(investment="fixed_real"),
                         cge.Scenario(investment_level=-1.0))
    with pytest.raises(ValueError, match="nonnegative"):
        cge.solve_period(econ, cge.Closure(investment="fixed_real"),
                         cge.Scenario(investment_level=np.nan))
