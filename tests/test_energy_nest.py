"""Value-added-energy nesting: fuel use that answers to prices."""

import numpy as np
import pandas as pd
import pytest

import cgelab as ck
from cgelab.examples import toy

FUEL = ["c_agr", "c_ind"]


def margins(**spec):
    _, econ = toy.build_with_margins(model_spec=toy.spec(energy=FUEL, **spec))
    inter = pd.DataFrame(0.0, index=econ.coms, columns=econ.acts)
    inter.loc[FUEL, ["a_agr", "a_ind"]] = [[2.0, 1.0], [4.0, 9.0]]
    return ck.add_emissions(econ, intermediate=inter)


PLAIN = (lambda e: ck.add_emissions(e, intermediate=margins().emissions.intermediate
                                    * e.io * e.X0[None, :]))(toy.build_with_margins()[1])
SHOCKS = [ck.Scenario(carbon_price=0.8), ck.Scenario(product_tax=PLAIN.ptax * 1.3),
          ck.Scenario(productivity=np.array([1.03, 1.0, 1.0])),
          ck.Scenario(carbon_price=0.5, world_price=np.array([1.2, 1.0, 1.0]))]


@pytest.mark.parametrize("sc", SHOCKS, ids=["carbon", "tax", "tfp", "mixed"])
def test_zero_elasticities_are_the_leontief_model(sc):
    nest = margins(sigma_vae=0.0, sigma_energy=0.0)
    a, b = ck.solve_period(PLAIN, scenario=sc), ck.solve_period(nest, scenario=sc)
    np.testing.assert_allclose(b.X, a.X, rtol=1e-12)
    np.testing.assert_allclose(b.pq, a.pq, rtol=1e-12)
    np.testing.assert_allclose(b.factor_prices, a.factor_prices, rtol=1e-12)
    assert ck.emissions(b)["total"] == pytest.approx(ck.emissions(a)["total"], rel=1e-12)


@pytest.mark.parametrize("spec", [dict(sigma_vae=0.4, sigma_energy=2.0),
                                  dict(sigma_vae=1.0, sigma_energy=1.0),
                                  dict(sigma_vae={"__default__": 0.6, "a_ind": 1.5},
                                       sigma_energy={"__default__": 2.0, "a_ind": 0.5})])
def test_base_year_homogeneity_and_walras(spec):
    econ = margins(**spec)
    assert ck.base_year_reproduction(econ)["ok"]
    assert ck.homogeneity_test(econ)["ok"]
    r = ck.solve_period(econ, scenario=ck.Scenario(carbon_price=0.8))
    assert abs(r.walras) < 1e-9
    na = ck.national_accounts(r)
    assert abs(na["gap"]) < 1e-9 * na["gdp_expenditure"]


def test_the_fuel_mix_satisfies_the_ces_first_order_conditions():
    """Fuel ratios answer to the fuel prices each activity pays, carbon included,
    and value added against energy to the bundle prices, by the calibrated shares."""
    sv, se = 0.4, 2.0
    econ = margins(sigma_vae=sv, sigma_energy=se)
    price = 0.8
    r = ck.solve_period(econ, scenario=ck.Scenario(carbon_price=price))
    nest, em = econ.energy_nest, econ.emissions
    p0 = 1.0 + econ.ptax
    for j in (0, 1):                                       # the two activities that burn fuel
        rel = (r.pqf + price * em.intermediate[:, j]) / p0
        q = r.input_coefficients[:, j] * p0                # in base-year value
        a, b = econ.commodity("c_agr"), econ.commodity("c_ind")
        expect = (nest.theta[a, j] / nest.theta[b, j]) * (rel[b] / rel[a]) ** se
        assert q[a] / q[b] == pytest.approx(expect, rel=1e-10)
    # Carbon raises the price of the dirtier fuel by more, so its share falls.
    base = ck.solve_period(econ)
    j = econ.activity("a_ind")
    share = lambda res: res.input_coefficients[a, j] / res.input_coefficients[:, j].sum()  # noqa
    assert share(r) != pytest.approx(share(base))


def test_more_substitution_cuts_more_emissions():
    totals = []
    for s in (0.0, 0.5, 1.5):
        econ = margins(sigma_vae=s, sigma_energy=s)
        totals.append(ck.emissions(ck.solve_period(econ, scenario=ck.Scenario(
            carbon_price=1.0)))["total"])
    assert totals[0] > totals[1] > totals[2]


def test_carbon_revenue_and_welfare_identity_under_the_nest():
    econ = margins(sigma_vae=0.6, sigma_energy=2.0)
    r = ck.solve_period(econ, scenario=ck.Scenario(carbon_price=0.8))
    assert r.government["carbon_production"] == pytest.approx(
        0.8 * ck.emissions(r)["intermediate"], rel=1e-12)
    w = ck.welfare_decomposition(econ, ck.Scenario(), ck.Scenario(
        carbon_price=0.8, productivity=np.array([1.02, 1.0, 1.0])))
    assert abs(w.residual) < 1e-8 and w.allocative_carbon_tax < 0


def test_fixed_proportion_activities_keep_their_fuel_mix():
    _, econ = toy.build_with_margins(model_spec=toy.spec(energy=FUEL, value_added="leontief",
                                                         sigma_vae=2.0, sigma_energy=2.0))
    assert np.all(econ.energy_nest.sigma_vae == 0) and np.all(econ.energy_nest.sigma_energy == 0)


def test_misuse_is_refused():
    with pytest.raises(KeyError, match="energy commodities"):
        toy.build(model_spec=toy.spec(energy=["c_oil"]))
    from cgelab.examples import regions
    with pytest.raises(ValueError, match="energy nest"):
        regions.build(model_spec=regions.spec(energy=["r1:c"]))


# ---------------------------------------------------------------------------
# Capital-energy composite inside value added (GTAP-E's arrangement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.7])
def test_equal_elasticities_collapse_to_the_value_added_nest(sigma):
    """With sigma_ke = sigma_va, the two-level nest is one CES over the other
    factors, capital and energy; so is the value-added nest with
    sigma_vae = sigma_va. The two must solve to the same equilibrium."""
    sc = ck.Scenario(carbon_price=0.8, productivity=np.array([1.02, 1.0, 1.0]))
    vae = margins(sigma_va=sigma, sigma_vae=sigma, sigma_energy=2.0)
    cap = margins(energy_nest="capital", sigma_va=sigma, sigma_ke=sigma, sigma_energy=2.0)
    a, b = ck.solve_period(vae, scenario=sc), ck.solve_period(cap, scenario=sc)
    np.testing.assert_allclose(b.X, a.X, rtol=1e-11)
    np.testing.assert_allclose(b.factor_prices, a.factor_prices, rtol=1e-11)
    np.testing.assert_allclose(b.input_coefficients, a.input_coefficients, rtol=1e-11, atol=1e-14)


def test_capital_and_energy_meet_their_first_order_condition():
    from cgelab.solve import _energy_index
    s_ke, s_e, price = 0.3, 2.0, 0.8
    econ = margins(energy_nest="capital", sigma_va=0.8, sigma_ke=s_ke, sigma_energy=s_e)
    r = ck.solve_period(econ, scenario=ck.Scenario(carbon_price=price))
    nest, k = econ.energy_nest, econ.factor("f_cap")
    p0 = 1.0 + econ.ptax
    for j in (0, 1):
        rel = (r.pqf + price * econ.emissions.intermediate[:, j]) / p0
        pe = _energy_index(nest.theta[:, j:j + 1], rel[:, None], np.array([s_e]))[0]
        energy = (r.input_coefficients[nest.mask, j] * p0[nest.mask] * rel[nest.mask]).sum() / pe
        capital = r.factor_demand_by_activity[k, j] / r.X[j]
        sk = nest.k_share[j]
        assert capital / energy == pytest.approx(
            sk / (1 - sk) * (pe / r.factor_prices[k]) ** s_ke, rel=1e-12)


def test_capital_energy_identities():
    econ = margins(energy_nest="capital", sigma_va=0.8, sigma_ke=0.3, sigma_energy=2.0)
    assert ck.base_year_reproduction(econ)["ok"]
    assert ck.homogeneity_test(econ)["ok"]
    r = ck.solve_period(econ, scenario=ck.Scenario(carbon_price=0.8))
    assert abs(r.walras) < 1e-9
    assert abs(ck.national_accounts(r)["gap"]) < 1e-9 * 200
    assert r.government["carbon_production"] == pytest.approx(
        0.8 * ck.emissions(r)["intermediate"], rel=1e-12)
    w = ck.welfare_decomposition(econ, ck.Scenario(), ck.Scenario(
        carbon_price=0.8, productivity=np.array([1.02, 1.0, 1.0])))
    assert abs(w.residual) < 1e-8


def test_energy_capital_must_be_a_factor():
    with pytest.raises(KeyError, match="energy_capital"):
        margins(energy_nest="capital", energy_capital="f_oil")
