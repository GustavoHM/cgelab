"""Public accounts and trade-price fields of regional results."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions
from tests.test_regional_institutions import build_institutional
from tests.test_regional_government import labelled_institutions


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
    assert financed == pytest.approx(result.savings)


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


@pytest.mark.parametrize("export_rule", ["fixed_shares", "cet"])
def test_world_currency_prices_and_accounts_survive_numeraire_change(export_rule):
    _, econ = regions.build(regions.spec(export_rule=export_rule))
    result = cge.solve_period(econ, cge.Closure(numeraire=2.0))
    np.testing.assert_allclose(result.world_import_price, [1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(result.world_export_price, [1.0, 1.0], atol=1e-9)
    accounts = cge.national_accounts(result)
    assert accounts["gdp_expenditure"] == pytest.approx(400.0)
    assert abs(accounts["gap"]) < 1e-8


@pytest.mark.parametrize("export_rule", ["fixed_shares", "cet"])
def test_trade_prices_preserve_shocks_and_origin_prices(export_rule):
    _, econ = regions.build(regions.spec(export_rule=export_rule))
    scenario = cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.2]),
                            export_price=np.array([1.3]) if export_rule == "cet" else None)
    result = cge.solve_period(econ, scenario=scenario)
    np.testing.assert_allclose(result.world_import_price, [1.2, 1.2], atol=1e-9)
    np.testing.assert_allclose(result.world_export_price * result.exchange_rate, result.pe,
                               rtol=1e-9)
    if export_rule == "cet":
        np.testing.assert_allclose(result.world_export_price, [1.3, 1.3], atol=1e-9)
    else:
        assert not np.isclose(result.world_export_price[0], result.world_export_price[1])
    assert abs(cge.national_accounts(result)["gap"]) < 1e-8


def test_accounts_close_with_transport_and_different_destination_prices():
    _, econ = regions.build_with_transport()
    tfp = np.ones(econ.n_act)
    tfp[0] = 1.1
    result = cge.solve_period(econ, scenario=cge.Scenario(productivity=tfp))
    assert abs(result.walras) < 1e-8
    assert abs(cge.national_accounts(result)["gap"]) < 1e-8


def test_household_price_index_matches_cobb_douglas_and_national_model():
    national, regional = labelled_institutions()
    scenario = cge.Scenario(world_price=np.array([1.0, 1.5]))
    ratios = []
    for econ in (national, regional):
        base = cge.solve_period(econ)
        alt = cge.solve_period(econ, scenario=scenario)
        ratio = alt.real["hh_price_index"] / base.real["hh_price_index"]
        exact = np.exp((econ.hh_budget * np.log(alt.pq / base.pq)).sum(axis=1))
        np.testing.assert_allclose(ratio, exact, rtol=1e-9)
        ratios.append(ratio)
    np.testing.assert_allclose(*ratios, rtol=1e-9)


@pytest.mark.parametrize("field, value", [
    ("government_absorbs", 10.0),
    ("tariff", np.array([0.5])),
    ("output_tax", np.array([0.5])),
    ("carbon_price", 10.0),
    ("pool_supply", {"r1:f_cap": np.array([50.0])}),
    ("labour_force", {"workers": 100.0}),
])
def test_unsupported_scenario_fields_are_refused(field, value):
    _, econ = regions.build()
    with pytest.raises(ValueError, match=field):
        cge.solve_period(econ, scenario=cge.Scenario(**{field: value}))


def _shocked_transport():
    _, econ = regions.build_with_transport()
    tfp = np.ones(econ.n_act)
    tfp[0] = 1.1
    return econ, cge.Scenario(productivity=tfp)


@pytest.mark.parametrize("case", ["transport", "regions"])
def test_indices_money_and_real_gdp_agree_with_the_accounts(case):
    if case == "transport":
        econ, scenario = _shocked_transport()
    else:
        _, econ = regions.build()
        scenario = cge.Scenario(productivity=np.array([1.1, 1.0]))
    base = cge.solve_period(econ)
    alt = cge.solve_period(econ, scenario=scenario)
    indices = cge.price_indices(base)
    np.testing.assert_allclose(indices.drop(index="terms_of_trade").to_numpy(dtype=float),
                               1.0, atol=1e-12)
    money = cge.in_money(base, alt)
    for period, column in ((base, "reference"), (alt, "alternative")):
        accounts = cge.national_accounts(period)
        assert abs(accounts["gap"]) < 1e-8
        for item in ("consumption", "government", "investment", "exports", "imports"):
            assert money.loc[("final_demand", item), column] == pytest.approx(accounts[item])
            value = cge.price_indices(period).loc[item, "value"]
            assert value * cge.national_accounts(base)[item] == pytest.approx(accounts[item])
        assert money.loc[("final_demand", "gdp"), column] == pytest.approx(
            accounts["gdp_expenditure"])
        real = cge.real_gdp(period)
        assert money.loc[("final_demand", "gdp"), f"{column}_base_prices"] == pytest.approx(
            real["expenditure"])
    assert abs(cge.real_gdp(base)["other"]) < 1e-9
    households = money.xs("households")
    np.testing.assert_allclose(households["reference_base_prices"], households["reference"])


def test_household_index_and_money_levels_match_the_national_model():
    national, regional = labelled_institutions()
    scenario = cge.Scenario(world_price=np.array([1.0, 1.5]))
    out = []
    for econ in (national, regional):
        base = cge.solve_period(econ)
        alt = cge.solve_period(econ, scenario=scenario)
        out.append((alt.real["hh_price_index"], alt.real["hh_real_income"],
                    cge.in_money(base, alt).to_numpy(dtype=float)))
    for a, b in zip(*out):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
