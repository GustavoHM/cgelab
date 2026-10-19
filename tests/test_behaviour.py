"""Behavioural microsimulation: occupational choice and earnings, linked to the model.

The estimators are checked against closed forms: a logit with intercepts only
gives the log ratios of the shares, and one with a single binary regressor the
log odds of each cell. The transition probabilities must add up to the logit
probabilities of both choices and agree with BRR's conditional draws. The link
must leave a survey alone when nothing changes and hit the model's employment
and earnings when something does.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab import micro
from cgelab.behaviour import Behaviour, PersonSurvey, conditional_draws, transition_probabilities
from cgelab.examples import toy

RNG = np.random.default_rng(20261019)


def dual(real_wage):
    return cge.Closure(labour_markets={"labour": cge.LabourMarket(
        rigid={"f_for": real_wage}, informal="f_inf")})


def harris_todaro(real_wage):
    return cge.Closure(labour_markets={"labour": cge.LabourMarket(
        rigid={"f_for": real_wage}, informal="f_inf", unemployment="harris_todaro",
        unemployment_rate=0.1)})


# -- estimation ---------------------------------------------------------------

def choice_survey(n=300, binary=False):
    status = RNG.choice(["a", "b", "inactive"], size=n, p=[0.5, 0.3, 0.2])
    x = RNG.integers(0, 2, size=n).astype(float)
    frame = pd.DataFrame({"household": 0, "status": status, "earnings": 1.0 + x, "x": x})
    return PersonSurvey(frame, characteristics=["x"],
                        choice_characteristics=["x"] if binary else [])


def test_a_logit_with_intercepts_only_is_the_log_share_ratios():
    p = choice_survey()
    w = RNG.uniform(0.5, 2.0, size=len(p.frame))
    b = Behaviour.estimate(p, ["a", "b"], weights=w)
    st = p.frame["status"].to_numpy()
    W = {s: w[st == s].sum() for s in ("a", "b", "inactive")}
    assert b.base == "inactive"
    assert b.choice[b.statuses.index("a"), 0] == pytest.approx(np.log(W["a"] / W["inactive"]),
                                                               abs=1e-7)
    assert b.choice[b.statuses.index("b"), 0] == pytest.approx(np.log(W["b"] / W["inactive"]),
                                                               abs=1e-7)


def test_a_logit_with_one_binary_regressor_is_the_log_odds_of_each_cell():
    p = choice_survey(binary=True)
    b = Behaviour.estimate(p, ["a", "b"])
    st, x = p.frame["status"].to_numpy(), p.frame["x"].to_numpy()
    for s in ("a", "b"):
        k = b.statuses.index(s)
        for cell in (0.0, 1.0):
            m = x == cell
            odds = np.log((st[m] == s).sum() / (st[m] == "inactive").sum())
            assert b.choice[k, 0] + cell * b.choice[k, 1] == pytest.approx(odds, abs=1e-6)


def test_the_earnings_equation_is_weighted_least_squares():
    p = toy.persons()
    w = RNG.uniform(0.5, 2.0, size=len(p.frame))
    b = Behaviour.estimate(p, ["f_for", "f_inf"], weights=w)
    f = p.frame
    m = (f["status"] == "f_for").to_numpy()
    X = np.column_stack([np.ones(m.sum()), f.loc[m, "education"], f.loc[m, "female"]])
    W = np.diag(w[m])
    coef = np.linalg.solve(X.T @ W @ X, X.T @ W @ np.log(f.loc[m, "earnings"]))
    np.testing.assert_allclose(b.earnings["f_for"]["coef"], coef, rtol=1e-10)


# -- transitions --------------------------------------------------------------

def test_transitions_add_up_to_both_choices():
    V = RNG.normal(size=(5, 4))
    delta = np.array([0.4, -0.7, 0.0, 1.2])
    T = transition_probabilities(V, delta)
    P0 = np.exp(V) / np.exp(V).sum(axis=1, keepdims=True)
    P1 = np.exp(V + delta) / np.exp(V + delta).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(T.sum(axis=2), P0, atol=1e-15)
    np.testing.assert_allclose(T.sum(axis=1), P1, atol=1e-15)
    still = transition_probabilities(V, np.zeros(4))
    np.testing.assert_allclose(still, np.einsum("ij,jk->ijk", P0, np.eye(4)), atol=1e-16)


def test_conditional_draws_keep_the_choice_and_average_to_the_transitions():
    V = np.tile(np.array([[0.2, -0.3, 0.5, 0.0]]), (200_000, 1))
    delta = np.array([0.6, 0.1, -0.4, 0.0])
    rng = np.random.default_rng(7)
    j = 2
    eps = conditional_draws(V, np.full(len(V), j), rng)
    assert np.all(np.argmax(V + eps, axis=1) == j)
    freq = np.bincount(np.argmax(V + delta + eps, axis=1), minlength=4) / len(V)
    T = transition_probabilities(V[:1], delta)[0]
    P0 = np.exp(V[0]) / np.exp(V[0]).sum()
    se = np.sqrt(freq * (1 - freq) / len(V))
    assert np.all(np.abs(freq - T[j] / P0[j]) < 4 * se + 1e-12)


# -- the link -----------------------------------------------------------------

def run(closure_alt, method="expected", **kw):
    _, econ = toy.build_with_segmented_labour()
    closure_ref = (harris_todaro(1.0) if closure_alt.labour_markets["labour"].unemployment
                   == "harris_todaro" else dual(1.0))
    ref, alt = cge.solve_period(econ, closure_ref), cge.solve_period(econ, closure_alt)
    res = micro.microsimulate(toy.survey(), ref, alt, persons=toy.persons(), method=method, **kw)
    return econ, ref, alt, res


@pytest.mark.parametrize("method", ["expected", "draws"])
def test_without_a_change_nobody_moves(method):
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ, dual(1.0))
    res = micro.microsimulate(toy.survey(), ref, ref, persons=toy.persons(), method=method,
                              replications=5)
    np.testing.assert_allclose(res.frame["real_alt"], res.frame["real_ref"], rtol=1e-12)
    w = res.workers
    np.testing.assert_allclose(w["alt"], w["ref"], rtol=1e-12)
    if method == "draws":
        base = np.repeat(res.frame["real_ref"].to_numpy()[:, None], 5, axis=1)
        np.testing.assert_allclose(res.replications, base, rtol=1e-12)


@pytest.mark.parametrize("closure", [dual(1.1), harris_todaro(1.02)], ids=["dual", "ht"])
def test_the_link_hits_the_models_employment_and_earnings(closure):
    econ, ref, alt, res = run(closure)
    cal = res.calibration
    np.testing.assert_allclose(cal["reached"], cal["target"], rtol=1e-10)
    m_ref, m_alt = ref.labour_markets["labour"], alt.labour_markets["labour"]
    for hi, h in enumerate(econ.hhs):
        row = cal.set_index(["group", "constraint"])
        for k, s in enumerate(["f_for", "f_inf"]):
            ratio = m_alt["household_employment"][hi, k] / m_ref["household_employment"][hi, k]
            assert row.at[(h, f"persons {s}"), "target"] == pytest.approx(
                ratio * row.at[(h, f"persons {s}"), "base"], rel=1e-12)
        if closure.labour_markets["labour"].unemployment == "harris_todaro":
            assert (h, "persons unemployed") in row.index
    # The micro labour income of every account is the model's, segment by segment.
    A_alt = micro.account_sources(alt)
    f = res.frame
    for h, s in itertools.product(econ.hhs, ["f_for", "f_inf"]):
        m = (f["group"] == h).to_numpy()
        assert f.loc[m, "weight"] @ res.income_alt.loc[m, s] == pytest.approx(A_alt.at[h, s],
                                                                              rel=1e-10)
    assert (res.frame["weight_alt"] == res.frame["weight"]).all()


def test_draws_are_reproducible_and_average_to_the_expected_labour_income():
    econ, ref, alt, expected = run(dual(1.1))
    _, _, _, a = run(dual(1.1), method="draws", replications=300, seed=11)
    _, _, _, b = run(dual(1.1), method="draws", replications=300, seed=11)
    np.testing.assert_array_equal(a.replications, b.replications)
    _, _, _, c = run(dual(1.1), method="draws", replications=300, seed=12)
    assert not np.array_equal(a.replications, c.replications)
    # Total real income, replication by replication, against the expected total.
    w = a.frame["weight"].to_numpy()
    totals = w @ a.replications
    se = totals.std(ddof=1) / np.sqrt(totals.size)
    assert abs(totals.mean() - w @ expected.frame["real_alt"]) < 4 * se
    ineq = micro.inequality(a)
    assert (ineq["alt_se"] > 0).all()
    pov = micro.poverty(a, 3.0)
    assert "alt_se" in pov.columns


def test_the_behavioural_and_the_reweighted_links_agree_on_the_models_totals():
    econ, ref, alt, behavioural = run(dual(1.1))
    reweighted = micro.microsimulate(toy.survey(), ref, alt)
    A_alt = micro.account_sources(alt)
    for res, weight in ((behavioural, "weight"), (reweighted, "weight_alt")):
        f = res.frame
        for h, s in itertools.product(econ.hhs, ["f_for", "f_inf", "f_cap"]):
            m = (f["group"] == h).to_numpy()
            assert f.loc[m, weight] @ res.income_alt.loc[m, s] == pytest.approx(
                A_alt.at[h, s], rel=1e-10)


def test_misuse_is_refused():
    _, plain = toy.build()
    ref = cge.solve_period(plain)
    with pytest.raises(ValueError, match="has none"):
        micro.microsimulate(toy.survey(), ref, ref, persons=toy.persons())
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ)
    p = toy.persons()
    lost = PersonSurvey(p.frame.assign(household=p.frame["household"] + 100),
                        characteristics=p.characteristics,
                        choice_characteristics=p.choice_characteristics)
    with pytest.raises(ValueError, match="not in the survey"):
        micro.microsimulate(toy.survey(), ref, ref, persons=lost)
    with pytest.raises(ValueError, match="method"):
        micro.microsimulate(toy.survey(), ref, ref, persons=p, method="simulated")
    with pytest.raises(ValueError, match="non-negative"):
        PersonSurvey(p.frame.assign(earnings=-1.0))
    with pytest.raises(ValueError, match="no person works"):
        Behaviour.estimate(p, ["f_for", "f_inf", "f_other"])
