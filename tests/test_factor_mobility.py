"""Imperfect factor mobility: a CET or logit transformation of an endowment across activities.

The checks that matter come from other code paths. At ω = 0, and with pools
that each reach one activity, the factor is a set of sector-specific factors,
which the solver already handles as separate accounts of a SAM written out by
hand. As ω grows the solution approaches the perfectly mobile model, at a rate
of 1/ω. The CET's income identity and the welfare decomposition close only if
the frontier, its price index and the envelope theorem are all written right.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.examples.toy import _BASE, _frame
from cgelab.mobility import FactorMobility

SHOCK = cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.0, 1.2]))
# Two pools of labour: one mostly in agriculture, one mostly in industry. The
# columns add up to the SAM's labour payments, 40 and 70.
POOLS = pd.DataFrame([[30.0, 10.0], [10.0, 60.0]], index=["rural", "urban"],
                     columns=["a_agr", "a_ind"])


def mobile(omega=2.0, form="cet", pools=None, sam=None, **spec):
    sp = toy.spec(numeraire_factor="f_cap", factor_mobility={"f_lab": omega},
                  factor_mobility_form=form, **spec)
    return cge.calibrate(sam or _frame(_BASE), sp,
                         factor_pools=None if pools is None else {"f_lab": pools})


def perfectly_mobile():
    return cge.calibrate(_frame(_BASE), toy.spec(numeraire_factor="f_cap"))


def specific_labour():
    """The toy with labour split by hand into one factor per activity.

    Each household owns the same share of both, its share of labour income in
    the toy (70/110 and 40/110), so the only difference from the toy is that
    labour cannot move.
    """
    drop = {("f_lab", "a_agr", 40.0), ("f_lab", "a_ind", 70.0),
            ("hh_1", "f_lab", 70.0), ("hh_2", "f_lab", 40.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [("f_la", "a_agr", 40.0), ("f_li", "a_ind", 70.0),
                ("hh_1", "f_la", 40 * 70 / 110), ("hh_2", "f_la", 40 * 40 / 110),
                ("hh_1", "f_li", 70 * 70 / 110), ("hh_2", "f_li", 70 * 40 / 110)]
    return cge.calibrate(_frame(entries), toy.spec(factors=("f_cap", "f_la", "f_li"),
                                                   numeraire_factor="f_cap"))


def same_equilibrium(a, b, tol=1e-9):
    for name in ("X", "Q", "pq", "hh_income", "imports", "exports"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=tol, atol=tol,
                                   err_msg=name)
    assert abs(a.exchange_rate - b.exchange_rate) < tol


CLOSURES = [cge.Closure(), cge.Closure(foreign="fixed_rate"),
            cge.Closure(investment="fixed_real"), cge.Closure(government="saving_rate"),
            cge.Closure(foreign_saving_unit="foreign_currency", government="nominal_fixed")]


@pytest.mark.parametrize("closure", CLOSURES, ids=range(len(CLOSURES)))
@pytest.mark.parametrize("form", ["cet", "logit"])
@pytest.mark.parametrize("pools", [None, POOLS], ids=["one_pool", "two_pools"])
def test_base_year_reproduces(closure, form, pools):
    econ = mobile(2.0, form, pools)
    rep = cge.base_year_reproduction(econ, closure)
    assert rep["ok"], rep["checks"]
    res = cge.solve_period(econ, closure)
    np.testing.assert_allclose(res.factor_prices_by_activity, 1.0, atol=1e-12)
    hom = cge.homogeneity_test(econ, closure)
    assert hom["ok"], hom["checks"]


@pytest.mark.parametrize("form", ["cet", "logit"])
def test_zero_elasticity_is_the_sector_specific_model(form):
    # Another code path: two separate factors in a SAM written by hand.
    fixed = cge.solve_period(mobile(0.0, form), scenario=SHOCK)
    split = cge.solve_period(specific_labour(), scenario=SHOCK)
    same_equilibrium(fixed, split)
    np.testing.assert_allclose(fixed.factor_prices_by_activity[0],
                               split.factor_prices[1:], rtol=1e-10)
    assert abs(fixed.walras) < 1e-12


@pytest.mark.parametrize("form", ["cet", "logit"])
def test_pools_that_reach_one_activity_each_are_sector_specific_at_any_elasticity(form):
    diagonal = pd.DataFrame([[40.0, 0.0], [0.0, 70.0]], index=["p_agr", "p_ind"],
                            columns=["a_agr", "a_ind"])
    res = cge.solve_period(mobile(3.0, form, diagonal), scenario=SHOCK)
    same_equilibrium(res, cge.solve_period(specific_labour(), scenario=SHOCK))


@pytest.mark.parametrize("form", ["cet", "logit"])
def test_one_pool_table_is_the_model_without_pools(form):
    one = pd.DataFrame([[40.0, 70.0]], index=["all"], columns=["a_agr", "a_ind"])
    same_equilibrium(cge.solve_period(mobile(2.0, form, one), scenario=SHOCK),
                     cge.solve_period(mobile(2.0, form), scenario=SHOCK), tol=1e-11)


@pytest.mark.parametrize("form", ["cet", "logit"])
def test_large_elasticity_approaches_perfect_mobility_at_rate_one_over_omega(form):
    ref = cge.solve_period(perfectly_mobile(), scenario=SHOCK)
    dist = {}
    for om in (1e3, 1e4):
        res = cge.solve_period(mobile(om, form), scenario=SHOCK)
        dist[om] = np.max(np.abs(res.X - ref.X))
        spread = np.ptp(res.factor_prices_by_activity[0])
        assert spread < 10.0 / om
    assert dist[1e4] < 1e-3
    assert dist[1e3] / dist[1e4] == pytest.approx(10.0, rel=0.01)


def test_wage_dispersion_falls_with_the_elasticity():
    spreads = [np.ptp(cge.solve_period(mobile(om), scenario=SHOCK).factor_prices_by_activity[0])
               for om in (0.0, 0.5, 2.0, 8.0)]
    assert all(a > b for a, b in zip(spreads, spreads[1:]))


def _transform(form, omega=1.7):
    delta = np.array([[0.2, 0.5, 0.3], [0.6, 0.0, 0.4]])
    return FactorMobility(factor="f", index=0, form=form, omega=omega,
                          users=np.ones(3, dtype=bool), pools=["a", "b"], delta=delta,
                          endowment0=np.array([10.0, 5.0]))


def test_cet_pool_income_is_its_price_index_times_its_endowment():
    m = _transform("cet")
    rng = np.random.default_rng(7)
    for _ in range(5):
        w = rng.uniform(0.5, 2.0, 3)
        L = rng.uniform(1.0, 20.0, 2)
        S, W = m.supply(w, L)
        np.testing.assert_allclose(S @ w, W * L, rtol=1e-13)
        # The frontier: the CET aggregate of the supplies is the endowment.
        e = (1.0 + m.omega) / m.omega
        reach = m.delta > 0
        agg = np.array([(np.sum(m.delta[o, reach[o]] ** (1 - e) * S[o, reach[o]] ** e))
                        ** (1 / e) for o in range(2)])
        np.testing.assert_allclose(agg, L, rtol=1e-12)
        assert np.all(S[~reach] == 0.0)


def test_logit_keeps_the_head_count():
    m = _transform("logit")
    w = np.array([0.7, 1.3, 1.1])
    S, W = m.supply(w, np.array([10.0, 5.0]))
    np.testing.assert_allclose(S.sum(axis=1), [10.0, 5.0], rtol=1e-14)
    np.testing.assert_allclose(W * [10.0, 5.0], S @ w, rtol=1e-14)


def test_the_two_forms_agree_to_first_order():
    cet, logit = _transform("cet"), _transform("logit")
    v = np.array([0.3, -0.5, 0.2])
    L = np.array([10.0, 5.0])
    S0 = cet.supply(np.ones(3), L)[0]
    gaps = []
    for eps in (1e-2, 5e-3):
        w = 1.0 + eps * v
        moved = np.max(np.abs(cet.supply(w, L)[0] - S0))
        gaps.append(np.max(np.abs(cet.supply(w, L)[0] - logit.supply(w, L)[0])) / moved)
    # The gap is second order, so relative to the first-order move it halves.
    assert gaps[0] / gaps[1] == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize("form", ["cet", "logit"])
@pytest.mark.parametrize("pools", [None, POOLS], ids=["one_pool", "two_pools"])
def test_walras_income_and_homogeneity_under_a_shock(form, pools):
    econ = mobile(2.0, form, pools)
    res = cge.solve_period(econ, scenario=SHOCK)
    assert abs(res.walras) < 1e-12
    # Factor income three ways: by activity at each activity's price, by pool,
    # and as the reported price times the reported endowment.
    by_activity = float((res.factor_prices_by_activity * res.factor_demand_by_activity).sum())
    pool = res.factor_pools["f_lab"]
    by_pool = float(pool["price"] @ pool["endowment"]) + float(
        res.factor_prices[1] * res.factor_demand[1])
    assert by_activity == pytest.approx(by_pool, rel=1e-12)
    assert float(res.factor_prices @ res.factor_demand) == pytest.approx(by_activity, rel=1e-12)
    # The pools clear each activity's labour market.
    np.testing.assert_allclose(pool["supply"].sum(axis=0), res.factor_demand_by_activity[0],
                               rtol=1e-10)
    twice = cge.solve_period(econ, cge.Closure(numeraire=2.0), SHOCK)
    np.testing.assert_allclose(twice.X, res.X, rtol=1e-10)
    np.testing.assert_allclose(twice.factor_prices_by_activity,
                               2.0 * res.factor_prices_by_activity, rtol=1e-10)


def test_logit_and_cet_equilibria_part_at_second_order():
    def gap(scale):
        sc = cge.Scenario(productivity=np.array([1.0 + scale, 1.0]))
        a = cge.solve_period(mobile(2.0, "cet"), scenario=sc)
        b = cge.solve_period(mobile(2.0, "logit"), scenario=sc)
        base = cge.solve_period(mobile(2.0, "cet"))
        moved = np.max(np.abs(a.factor_prices_by_activity - base.factor_prices_by_activity))
        return np.max(np.abs(a.factor_prices_by_activity - b.factor_prices_by_activity)) / moved
    assert gap(0.02) / gap(0.01) == pytest.approx(2.0, rel=0.05)


def test_welfare_decomposition_closes_and_prices_reallocation():
    alt = cge.Scenario(product_tax=toy.build()[1].ptax * 0.5, productivity=np.array([1.05, 1.0]))
    cet = cge.welfare_decomposition(mobile(2.0, "cet", POOLS), cge.Scenario(), alt)
    logit = cge.welfare_decomposition(mobile(2.0, "logit", POOLS), cge.Scenario(), alt)
    assert abs(cet["residual"]) < 1e-8 and abs(logit["residual"]) < 1e-8
    # The envelope theorem on the CET pool's revenue: moving costs exactly what
    # the wage gaps pay, so there is no gain left over.
    assert abs(cet["factor_reallocation"]) < 1e-8
    assert logit["factor_reallocation"] > 1e-6
    plain = cge.welfare_decomposition(perfectly_mobile(), cge.Scenario(), alt)
    assert plain["factor_reallocation"] == 0.0


def test_moving_endowment_between_pools():
    econ = mobile(2.0, "cet", POOLS)
    base = cge.solve_period(econ)
    same = cge.solve_period(econ, scenario=cge.Scenario(pool_supply={"f_lab": [40.0, 70.0]}))
    np.testing.assert_allclose(same.factor_prices_by_activity, 1.0, atol=1e-12)
    # Ten workers leave the rural pool for the urban one: more labour reaches
    # industry, whose wage falls relative to agriculture's.
    res = cge.solve_period(econ, scenario=cge.Scenario(pool_supply={"f_lab": [30.0, 80.0]}))
    wa, wi = res.factor_prices_by_activity[0]
    assert wi / wa < 1.0
    assert res.factor_demand_by_activity[0, 1] > base.factor_demand_by_activity[0, 1]
    assert abs(res.walras) < 1e-12
    # Pools scale with the factor's supply when they are not given.
    grown = cge.solve_period(econ, scenario=cge.Scenario(
        factor_supply=econ.fac_supply * np.array([1.1, 1.0])))
    np.testing.assert_allclose(grown.factor_pools["f_lab"]["endowment"], [44.0, 77.0])


def test_a_path_with_labour_growth():
    econ = mobile(2.0, "cet", POOLS)
    path = cge.run_path(econ, years=range(2026, 2031),
                        growth=cge.Growth(labour=0.02, tfp=0.01))
    for p in path.periods:
        assert p.converged and abs(p.walras) < 1e-9
    last = path.periods[-1].factor_pools["f_lab"]["endowment"]
    np.testing.assert_allclose(last.sum(), 110.0 * 1.02 ** 4, rtol=1e-12)


def test_capital_energy_nest_with_imperfectly_mobile_capital():
    _, econ = toy.build_with_margins(model_spec=toy.spec(
        energy=["c_agr", "c_ind"], energy_nest="capital", sigma_ke=0.4, sigma_energy=2.0,
        numeraire_factor="f_lab", factor_mobility={"f_cap": 1.5}))
    rep = cge.base_year_reproduction(econ)
    assert rep["ok"], rep["checks"]
    res = cge.solve_period(econ, scenario=cge.Scenario(product_tax=econ.ptax * 1.5))
    assert abs(res.walras) < 1e-12
    assert np.ptp(res.factor_prices_by_activity[1][econ.mobility[1].users]) > 1e-6


def test_misuse_is_refused():
    with pytest.raises(ValueError, match="numeraire"):
        toy.spec(factor_mobility={"f_lab": 1.0})
    with pytest.raises(ValueError, match=">= 0"):
        toy.spec(numeraire_factor="f_cap", factor_mobility={"f_lab": -1.0})
    with pytest.raises(ValueError, match="perfectly mobile factor is one left out"):
        toy.spec(numeraire_factor="f_cap", factor_mobility={"f_lab": float("inf")})
    with pytest.raises(ValueError, match="not among"):
        toy.spec(numeraire_factor="f_cap", factor_mobility={"f_land": 1.0})
    with pytest.raises(ValueError, match="'cet' or 'logit'"):
        toy.spec(numeraire_factor="f_cap", factor_mobility={"f_lab": 1.0},
                 factor_mobility_form="ces")
    with pytest.raises(ValueError, match="regional"):
        cge.ModelSpec(regions=("r1",), numeraire_factor="r1:f_cap",
                      factor_mobility={"f_lab": 1.0})
    bad = POOLS.copy()
    bad.loc["rural", "a_agr"] = 31.0
    with pytest.raises(ValueError, match="does not add up"):
        mobile(2.0, pools=bad)
    with pytest.raises(ValueError, match="does not list"):
        cge.calibrate(_frame(_BASE), toy.spec(numeraire_factor="f_cap"),
                      factor_pools={"f_lab": POOLS})
    econ = mobile(2.0, pools=POOLS)
    with pytest.raises(ValueError, match="only 'full_employment'"):
        cge.solve_period(econ, cge.Closure(labour="wage_curve", natural_unemployment=0.05))
    with pytest.raises(ValueError, match="one entry per pool"):
        cge.solve_period(econ, scenario=cge.Scenario(pool_supply={"f_lab": [110.0]}))
    with pytest.raises(ValueError, match="agree|same total"):
        cge.solve_period(econ, scenario=cge.Scenario(
            pool_supply={"f_lab": [40.0, 80.0]}, factor_supply=econ.fac_supply))
    with pytest.raises(ValueError, match="not imperfectly"):
        cge.solve_period(econ, scenario=cge.Scenario(pool_supply={"f_cap": [60.0]}))


def test_perfectly_mobile_models_report_one_price_everywhere():
    res = cge.solve_period(perfectly_mobile(), scenario=SHOCK)
    assert res.factor_pools is None
    np.testing.assert_allclose(res.factor_prices_by_activity,
                               np.tile(res.factor_prices[:, None], (1, 2)), rtol=0)
    unchanged = dataclasses.replace(perfectly_mobile().spec)
    assert unchanged.factor_mobility is None
