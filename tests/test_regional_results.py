"""Public accounts and trade-price fields of regional results."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions
from tests.test_regional_institutions import build_institutional


def test_government_account_closes_against_regional_savings():
    _, econ = regions.build()
    result = cge.solve_period(econ)
    account = cge.government_account(result)
    assert account["revenue"] == pytest.approx(20.0)
    assert account["consumption"] == pytest.approx(20.0)
    assert account["saving"] == pytest.approx(0.0, abs=1e-10)
    assert account["saving"] == pytest.approx(result.real["government_saving"])


def test_government_account_uses_the_same_fixed_real_financing():
    econ = build_institutional()
    result = cge.solve_period(econ, cge.Closure(investment="fixed_real"),
                              cge.Scenario(investment_level=80.0))
    account = cge.government_account(result)
    assert account["saving"] == pytest.approx(result.real["government_saving"])
    financed = (result.household_saving + result.forced_saving
                + account["saving"] + result.foreign_saving)
    assert financed == result.savings


def test_national_accounts_and_trade_prices_are_populated():
    _, econ = regions.build()
    result = cge.solve_period(econ)
    accounts = cge.national_accounts(result)
    assert accounts["gdp_expenditure"] == pytest.approx(200.0)
    assert accounts["gdp_income"] == pytest.approx(200.0)
    assert abs(accounts["gap"]) < 1e-8
    np.testing.assert_allclose(result.world_export_price, [1.0, 1.0])
    np.testing.assert_allclose(result.world_import_price, [1.0, 1.0])
    np.testing.assert_allclose(result.pe, [1.0, 1.0])
    np.testing.assert_allclose(result.domestic_sales, [90.0, 90.0])


def test_transport_has_a_complete_government_account():
    _, econ = regions.build_with_transport()
    result = cge.solve_period(econ)
    account = cge.government_account(result)
    assert account["revenue"] == pytest.approx(20.0)
    assert account["consumption"] == pytest.approx(20.0)
    assert abs(account["saving"]) < 1e-8
    assert result.factor_income_abroad == 0.0
