"""Segmented labour: formal and informal accounts that share one population.

The toy's labour is split by hand into a formal account, used by industry, and
an informal one, used by agriculture, each household owning the same share of
both. With no rigid segment the group is one labour market, and it must give
back the toy with one mobile labour factor, which the solver reaches by another
code path. A dual market whose rigid wage is set where that market would put it
must not ration anyone. Harris–Todaro must keep its own identity, and with that
same rigid wage it must approach the one market as base-year unemployment goes
to zero.
"""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.examples.toy import _BASE, _frame

SHOCK = cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.0, 1.2]))
GROUP = {"labour": ["f_for", "f_inf"]}


def split_labour(**spec):
    """The toy with labour split into a formal (industry) and an informal (farm) account."""
    drop = {("f_lab", "a_agr", 40.0), ("f_lab", "a_ind", 70.0),
            ("hh_1", "f_lab", 70.0), ("hh_2", "f_lab", 40.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [("f_inf", "a_agr", 40.0), ("f_for", "a_ind", 70.0),
                ("hh_1", "f_inf", 40 * 70 / 110), ("hh_2", "f_inf", 40 * 40 / 110),
                ("hh_1", "f_for", 70 * 70 / 110), ("hh_2", "f_for", 70 * 40 / 110)]
    return cge.calibrate(_frame(entries), toy.spec(factors=("f_cap", "f_for", "f_inf"),
                                                   numeraire_factor="f_cap", **spec))


def unequal_ownership():
    """Formal labour owned mostly by hh_1, informal mostly by hh_2 (the bundled toy)."""
    return toy.build_with_segmented_labour()[1]


def dual(real_wage=1.0):
    return cge.Closure(labour_markets={"labour": cge.LabourMarket(
        rigid={"f_for": real_wage}, informal="f_inf")})


def harris_todaro(rate=0.1, real_wage=1.0, **closure):
    return cge.Closure(labour_markets={"labour": cge.LabourMarket(
        rigid={"f_for": real_wage}, informal="f_inf", unemployment="harris_todaro",
        unemployment_rate=rate)}, **closure)


def same_equilibrium(a, b, tol=1e-9):
    for name in ("X", "Q", "pq", "hh_income", "imports", "exports"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=tol, atol=tol,
                                   err_msg=name)


MARKETS = {"common": cge.Closure(), "dual": dual(), "harris_todaro": harris_todaro()}
OTHER = [dict(), dict(foreign="fixed_rate"), dict(investment="fixed_real"),
         dict(government="saving_rate")]


@pytest.mark.parametrize("other", OTHER, ids=range(len(OTHER)))
@pytest.mark.parametrize("market", list(MARKETS))
def test_base_year_homogeneity_and_walras(market, other):
    import dataclasses
    closure = dataclasses.replace(MARKETS[market], **other)
    for econ in (split_labour(labour_segments=GROUP), unequal_ownership()):
        rep = cge.base_year_reproduction(econ, closure)
        assert rep["ok"], rep["checks"]
        hom = cge.homogeneity_test(econ, closure)
        assert hom["ok"], hom["checks"]
        res = cge.solve_period(econ, closure, SHOCK)
        assert abs(res.walras) < 1e-12
        # A fixed exchange rate is in numeraire units too, and doubles with it.
        twice = cge.solve_period(econ, dataclasses.replace(
            closure, numeraire=2.0, exchange_rate=2.0 * closure.exchange_rate), SHOCK)
        np.testing.assert_allclose(twice.X, res.X, rtol=1e-9)
        np.testing.assert_allclose(twice.factor_prices, 2.0 * res.factor_prices, rtol=1e-9)


def test_one_labour_market_is_the_toy_with_mobile_labour():
    # Another code path: the toy's single labour account, perfectly mobile.
    group = cge.solve_period(split_labour(labour_segments=GROUP), scenario=SHOCK)
    toy_res = cge.solve_period(cge.calibrate(_frame(_BASE), toy.spec(numeraire_factor="f_cap")),
                               scenario=SHOCK)
    same_equilibrium(group, toy_res, tol=1e-10)
    lab = toy_res.factor_prices[toy_res.econ.factor("f_lab")]
    np.testing.assert_allclose(group.factor_prices[1:], lab, rtol=1e-10)


def test_a_rigid_wage_that_does_not_bind_rations_nobody():
    econ = unequal_ownership()
    common = cge.solve_period(econ, scenario=SHOCK)
    target = float(common.labour_markets["labour"]["real_wage"][0])
    rationed = cge.solve_period(econ, dual(target), SHOCK)
    same_equilibrium(rationed, common)
    np.testing.assert_allclose(rationed.labour_markets["labour"]["household_employment"],
                               common.labour_markets["labour"]["household_employment"],
                               atol=1e-9)


def test_dual_market_accounts():
    econ = unequal_ownership()
    res = cge.solve_period(econ, dual(1.05), SHOCK)
    m = res.labour_markets["labour"]
    # The rigid wage is the target times the purchaser-price index.
    assert m["real_wage"][0] == pytest.approx(1.05, rel=1e-12)
    # Everyone works: employment adds up to the labour force, which is the base year's.
    assert m["unemployment"] == 0.0
    assert m["employment"].sum() == pytest.approx(110.0, rel=1e-12)
    # Households: workforce fixed, formal jobs in base-year shares, the rest informal.
    e = m["household_employment"]
    np.testing.assert_allclose(e.sum(axis=0), m["employment"], rtol=1e-12)
    np.testing.assert_allclose(e.sum(axis=1), m["household_workforce"], rtol=1e-12)
    np.testing.assert_allclose(m["household_workforce"], [70.0, 40.0], rtol=1e-12)
    np.testing.assert_allclose(e[:, 0], np.array([60.0, 10.0]) / 70.0 * m["employment"][0],
                               rtol=1e-12)
    # A higher formal wage hires fewer formal workers, and they go informal.
    lower = cge.solve_period(econ, dual(1.0), SHOCK).labour_markets["labour"]
    assert m["employment"][0] < lower["employment"][0]
    assert m["employment"][1] > lower["employment"][1]


def test_lost_formal_jobs_move_informal_income_to_the_same_household():
    econ = unequal_ownership()
    res = cge.solve_period(econ, dual(1.2), cge.Scenario())
    e = res.labour_markets["labour"]["household_employment"]
    lost = np.array([60.0, 10.0]) - e[:, 0]
    np.testing.assert_allclose(e[:, 1] - np.array([10.0, 30.0]), lost, rtol=1e-10)
    # Labour income by household is each household's workers at their wages.
    w = res.factor_prices[1:]
    labour = e @ w
    capital = econ.hh_facshare[:, 0] * res.factor_prices[0] * res.factor_demand[0]
    np.testing.assert_allclose(res.hh_income, labour + capital, rtol=1e-12)


def test_harris_todaro_identity_and_accounts():
    econ = unequal_ownership()
    base = cge.solve_period(econ, harris_todaro(0.1))
    m0 = base.labour_markets["labour"]
    assert m0["unemployment_rate"] == pytest.approx(0.1, rel=1e-12)
    assert m0["labour_force"] == pytest.approx(110.0 + 70.0 / 9.0, rel=1e-12)
    res = cge.solve_period(econ, harris_todaro(0.1, real_wage=1.1), SHOCK)
    m = res.labour_markets["labour"]
    E_F, E_I = m["employment"]
    U = m["unemployment"]
    assert E_F + E_I + U == pytest.approx(m["labour_force"], rel=1e-12)
    w_F, w_I = m["wage"]
    assert w_I == pytest.approx(w_F * E_F / (E_F + U) / 0.9, rel=1e-10)
    e = m["household_employment"]
    np.testing.assert_allclose(e.sum(axis=1) + m["household_unemployed"],
                               m["household_workforce"], rtol=1e-12)
    np.testing.assert_allclose(m["household_unemployed"].sum(), U, rtol=1e-12)
    # A higher formal wage lengthens the queue.
    assert U > m0["unemployment"]


def test_harris_todaro_with_vanishing_unemployment_is_one_labour_market():
    # Harris–Todaro does not tend to the dual market: it holds the informal wage
    # at the expected rigid wage, and the dual market lets any gap stand. With
    # the rigid wage where one labour market puts it and base-year unemployment
    # going to zero, the queue empties and the two coincide, at a rate of u0.
    econ = unequal_ownership()
    common = cge.solve_period(econ, scenario=SHOCK)
    target = float(common.labour_markets["labour"]["real_wage"][0])
    gaps = []
    for rate in (1e-3, 1e-4):
        h = cge.solve_period(econ, harris_todaro(rate, real_wage=target), SHOCK)
        gaps.append(np.max(np.abs(h.X - common.X)))
        assert h.labour_markets["labour"]["unemployment"] < 100 * rate
    assert gaps[0] / gaps[1] == pytest.approx(10.0, rel=0.02)


def test_labour_force_and_growth():
    econ = unequal_ownership()
    more = cge.solve_period(econ, dual(), cge.Scenario(labour_force={"labour": 121.0}))
    m = more.labour_markets["labour"]
    assert m["employment"].sum() == pytest.approx(121.0, rel=1e-12)
    np.testing.assert_allclose(m["household_workforce"], [77.0, 44.0], rtol=1e-12)
    path = cge.run_path(econ, dual(), years=range(2026, 2031),
                        growth=cge.Growth(labour=0.02, labour_factor="labour",
                                          capital_factor="f_cap"))
    for p in path.periods:
        assert p.converged and abs(p.walras) < 1e-9
    assert path.periods[-1].labour_markets["labour"]["labour_force"] == pytest.approx(
        110.0 * 1.02 ** 4, rel=1e-12)


@pytest.mark.parametrize("market", ["dual", "harris_todaro"])
def test_welfare_decomposition_closes(market):
    econ = unequal_ownership()
    alt = cge.Scenario(product_tax=econ.ptax * 0.5, productivity=np.array([1.05, 1.0]))
    d = cge.welfare_decomposition(econ, cge.Scenario(), alt, closure=MARKETS[market])
    assert abs(d["residual"]) < 1e-8


def test_misuse_is_refused():
    with pytest.raises(ValueError, match="two or more"):
        toy.spec(numeraire_factor="f_cap", labour_segments={"labour": ["f_lab"]})
    with pytest.raises(ValueError, match="numeraire"):
        toy.spec(labour_segments={"labour": ["f_lab", "f_cap"]})
    with pytest.raises(ValueError, match="also a factor"):
        toy.spec(numeraire_factor="f_cap", labour_segments={"f_lab": ["f_lab", "f_cap"]})
    with pytest.raises(ValueError, match="factor_mobility"):
        toy.spec(factors=("f_cap", "f_for", "f_inf"), numeraire_factor="f_cap",
                 labour_segments=GROUP, factor_mobility={"f_for": 1.0})
    with pytest.raises(ValueError, match="needs a rigid segment"):
        cge.LabourMarket(unemployment="harris_todaro", unemployment_rate=0.1)
    with pytest.raises(ValueError, match="strictly between"):
        cge.LabourMarket(rigid={"f_for": 1.0}, unemployment="harris_todaro")
    with pytest.raises(ValueError, match="read only under"):
        cge.LabourMarket(rigid={"f_for": 1.0}, unemployment_rate=0.1)
    with pytest.raises(ValueError, match="cannot also be rigid"):
        cge.LabourMarket(rigid={"f_inf": 1.0}, informal="f_inf")
    econ = unequal_ownership()
    with pytest.raises(ValueError, match="not groups"):
        cge.solve_period(econ, cge.Closure(labour_markets={"work": cge.LabourMarket()}))
    with pytest.raises(ValueError, match="not in the group"):
        cge.solve_period(econ, cge.Closure(labour_markets={"labour": cge.LabourMarket(
            rigid={"f_cap": 1.0})}))
    with pytest.raises(ValueError, match="labour segment"):
        cge.solve_period(econ, cge.Closure(labour="fixed_real_wage", labour_factor="f_for"))
    with pytest.raises(TypeError, match="LabourMarket"):
        cge.Closure(labour_markets={"labour": {"rigid": {"f_for": 1.0}}})
    # A rigid wage so low that formal demand exceeds a household's workforce.
    with pytest.raises((ValueError, RuntimeError)):
        cge.solve_period(econ, dual(0.2))
