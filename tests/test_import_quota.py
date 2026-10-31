"""Import quotas: quota − imports ≥ 0 ⊥ premium ≥ 0, the rent to the licence holders."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import hosoe, regions, toy


def test_a_slack_quota_changes_nothing():
    _, econ = toy.build()
    free = cge.solve_period(econ)
    slack = cge.solve_period(econ, scenario=cge.Scenario(import_quota={"c_agr": 30.0}))
    np.testing.assert_allclose(slack.X, free.X, rtol=1e-12)
    assert slack.quota["c_agr"]["premium"] == 0.0 and slack.quota["c_agr"]["rent"] == 0.0


@pytest.mark.parametrize("holder", ["government", "f_cap"])
def test_a_binding_quota_holds_imports_and_pays_its_rent(holder):
    _, econ = toy.build()
    result = cge.solve_period(econ, scenario=cge.Scenario(import_quota={"c_agr": 13.5},
                                                          quota_rent_to=holder))
    assert result.converged and abs(result.walras) < 1e-10
    quota = result.quota["c_agr"]
    assert result.imports[econ.commodity("c_agr")] == pytest.approx(13.5, rel=1e-12)
    assert quota["premium"] > 0.01
    border = result.exchange_rate * result.world_import_price[econ.commodity("c_agr")]
    assert quota["rent"] == pytest.approx(quota["premium"] * border * 13.5, rel=1e-12)
    accounts = cge.national_accounts(result)
    assert abs(accounts["gap"]) < 1e-8
    assert accounts["quota_rent"] == pytest.approx(quota["rent"])
    booked = cge.government_account(result)["quota_rent"]
    assert booked == pytest.approx(quota["rent"] if holder == "government" else 0.0)


def test_a_premium_can_jump_and_the_solver_still_finds_it():
    """Fixed export shares pin the import bill; a small cut needs a large premium."""
    _, econ = toy.build()
    result = cge.solve_period(econ, scenario=cge.Scenario(import_quota={"c_ind": 24.9}))
    assert result.converged and abs(result.walras) < 1e-10
    assert result.quota["c_ind"]["premium"] > 1.0
    assert result.imports[econ.commodity("c_ind")] == pytest.approx(24.9, rel=1e-12)


def test_import_quotas_are_homogeneous_of_degree_zero():
    _, econ = toy.build()
    scenario = cge.Scenario(import_quota={"c_agr": 13.5})
    one = cge.solve_period(econ, scenario=scenario)
    two = cge.solve_period(econ, cge.Closure(numeraire=2.0), scenario)
    np.testing.assert_allclose(two.X, one.X, rtol=1e-10)
    assert two.quota["c_agr"]["premium"] == pytest.approx(one.quota["c_agr"]["premium"])
    assert two.quota["c_agr"]["rent"] == pytest.approx(2.0 * one.quota["c_agr"]["rent"])


def test_import_quotas_are_refused_where_they_cannot_hold():
    _, econ = toy.build()
    with pytest.raises(KeyError, match="c_none"):
        cge.solve_period(econ, scenario=cge.Scenario(import_quota={"c_none": 1.0}))
    with pytest.raises(ValueError, match="nonnegative"):
        cge.solve_period(econ, scenario=cge.Scenario(import_quota={"c_agr": -1.0}))
    with pytest.raises(KeyError, match="f_land"):
        cge.solve_period(econ, scenario=cge.Scenario(import_quota={"c_agr": 13.5},
                                                     quota_rent_to="f_land"))
    with pytest.raises(ValueError, match="quota"):
        cge.to_jump(econ, scenario=cge.Scenario(import_quota={"c_agr": 13.5}))
    _, large = hosoe.build(large_country=True)
    with pytest.raises(ValueError, match="import-supply curve"):
        cge.solve_period(large, hosoe.CLOSURE, cge.Scenario(import_quota={"c_brd": 10.0}))
    _, regional = regions.build()
    with pytest.raises(ValueError, match="import_quota"):
        cge.solve_period(regional, scenario=cge.Scenario(import_quota={"r1:c": 5.0}))
