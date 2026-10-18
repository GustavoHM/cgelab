"""Calibration: units, margins, negative value added, own-account production."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy


def test_coefficients_are_in_basic_price_units():
    """Intermediates sit at purchaser prices in the SAM and must be deflated by (1+t).

    The check: the unit cost of every activity is exactly one at base-year
    prices. It is not if the tax is counted twice.
    """
    sam, econ = toy.build()
    res = cge.solve_period(econ)
    assert np.allclose(res.pa, 1.0, atol=1e-12)


def test_market_shares_sum_to_one_over_activities():
    sam, econ = toy.build_with_margins()
    produced = econ.Q0 > 0
    assert np.allclose(econ.msh[:, produced].sum(axis=0), 1.0, atol=1e-12)


def test_margin_block_is_read_and_its_diagonal_dropped():
    """The margin services must appear as a Leontief requirement, not vanish."""
    sam, econ = toy.build_with_margins()
    assert econ.mq.sum() > 0.0
    assert np.allclose(np.diag(econ.mq), 0.0)
    assert cge.base_year_reproduction(econ)["ok"]


def test_own_account_activity_is_detected_and_carried():
    sam, econ = toy.build_with_own_production()
    j = econ.activity("a_own")
    assert econ.own_account[j]
    assert not econ.own_account[econ.activity("a_agr")]
    res = cge.solve_period(econ)
    assert res.X[j] == pytest.approx(econ.X0[j], rel=1e-12)
    assert res.hh_direct_outlay.sum() == pytest.approx(9.0, rel=1e-12)


def test_negative_value_added_falls_back_to_fixed_proportions():
    """A negative operating surplus is real data; no CES nest accepts it."""
    import pandas as pd
    # a_agr keeps its value added of 65 but splits it 70 labour, -5 capital. The
    # factor totals then move by 30 each way, so household factor income moves
    # with them and every account still balances.
    drop = {("f_cap", "a_agr", 25.0), ("f_lab", "a_agr", 40.0),
            ("hh_1", "f_lab", 70.0), ("hh_1", "f_cap", 45.0)}
    entries = [e for e in toy._BASE if tuple(e) not in drop]
    entries += [("f_lab", "a_agr", 70.0), ("f_cap", "a_agr", -5.0),
                ("hh_1", "f_lab", 100.0), ("hh_1", "f_cap", 15.0)]
    sam = cge.SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]))
    assert sam.check_balance()["ok"]
    econ = cge.calibrate(sam, toy.spec())
    assert econ.leontief[econ.activity("a_agr")]
    assert "a_agr" in econ.diagnostics["negative_value_added"]
    assert cge.base_year_reproduction(econ)["ok"]


def test_les_calibration_gives_positive_subsistence():
    sam, econ = toy.build(model_spec=toy.spec(household_demand="les", frisch=-2.0))
    assert np.all(econ.hh_committed >= 0.0)
    assert econ.hh_committed.sum() > 0.0
    assert cge.base_year_reproduction(econ)["ok"]


def test_impossible_frisch_parameter_is_refused():
    with pytest.raises(ValueError, match="NEGATIVE subsistence"):
        toy.build(model_spec=toy.spec(household_demand="les", frisch=-0.5))
