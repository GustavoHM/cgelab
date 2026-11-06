"""SAM estimation by cross entropy, and location quotients."""

import math

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab import SAM, BalanceError
from cgelab.examples import toy

BASE = toy._frame(toy._BASE)


def _prior(seed=7, negative=False):
    f = BASE.frame.copy()
    f["value"] = f.value * np.random.default_rng(seed).uniform(0.8, 1.25, len(f))
    if negative:
        f.loc[(f.row == "s_i") & (f.col == "gov"), "value"] = -5.0
    return SAM(frame=f, name="prior")


def _ras(prior, totals, iters=20000):
    """Biproportional scaling to row and column totals, written out: the reference."""
    acc = prior.accounts
    X = prior.matrix(acc).to_numpy(dtype=float).copy()
    y = np.array([totals[a] for a in acc])
    for _ in range(iters):
        X *= (y / X.sum(axis=1))[:, None]
        X *= (y / X.sum(axis=0))[None, :]
    return X


def test_with_every_total_given_it_is_ras():
    prior = _prior()
    totals = BASE.receipts().to_dict()
    est = cge.cross_entropy(prior, totals=totals)
    acc = prior.accounts
    np.testing.assert_allclose(est.sam.matrix(acc).to_numpy(), _ras(prior, totals), atol=1e-10)


def test_the_estimate_balances_and_keeps_the_prior_zeros_and_signs():
    prior = _prior(negative=True)
    est = cge.cross_entropy(prior)
    assert est.sam.imbalance().abs().max() < 1e-10
    acc = prior.accounts
    a, b = est.sam.matrix(acc).to_numpy(), prior.matrix(acc).to_numpy()
    assert np.all((a == 0) == (b == 0))
    assert np.all(np.sign(a) == np.sign(b))
    assert est.sam.matrix(acc).loc["s_i", "gov"] < 0


def test_a_balanced_prior_is_its_own_estimate():
    est = cge.cross_entropy(BASE)
    np.testing.assert_allclose(est.sam.matrix(BASE.accounts).to_numpy(),
                               BASE.matrix().to_numpy(), atol=1e-10)
    assert est.divergence == pytest.approx(0.0, abs=1e-12)


def test_the_information_holds_and_has_shadow_values():
    prior = _prior()
    exports = ({("c_agr", "row"): 1.0, ("c_ind", "row"): 1.0}, 31.0)
    est = cge.cross_entropy(prior, totals={"a_agr": 100.0}, fixed=[("f_lab", "a_agr")],
                            targets={("c_agr", "hh_1"): 33.0}, aggregates=[exports])
    m = est.sam.matrix(prior.accounts)
    assert m.loc["a_agr"].sum() == pytest.approx(100.0, abs=1e-9)
    assert m.loc["f_lab", "a_agr"] == prior.matrix(prior.accounts).loc["f_lab", "a_agr"]
    assert m.loc["c_agr", "hh_1"] == pytest.approx(33.0, abs=1e-9)
    assert m.loc["c_agr", "row"] + m.loc["c_ind", "row"] == pytest.approx(31.0, abs=1e-9)
    assert list(est.information_multipliers.index) == [
        "total:a_agr", "cell:c_agr<-hh_1", "aggregate:0"]
    # More information can only move the estimate further from the prior.
    assert est.divergence > cge.cross_entropy(prior).divergence


def test_information_the_prior_cannot_meet_is_refused():
    prior = _prior()
    with pytest.raises(ValueError, match="leaves empty"):
        cge.cross_entropy(prior, targets={("f_lab", "row"): 1.0})
    with pytest.raises(KeyError, match="not an account"):
        cge.cross_entropy(prior, totals={"nowhere": 1.0})
    with pytest.raises(BalanceError):
        cge.cross_entropy(prior, totals={"a_agr": 100.0},
                          targets={("a_agr", "c_agr"): 90.0})   # a_agr's only receipt


def test_location_quotients_by_hand():
    reg = {"A": 10.0, "B": 30.0, "C": 60.0}
    nat = {"A": 200.0, "B": 300.0, "C": 500.0}
    slq = {"A": 0.5, "B": 1.0, "C": 1.2}
    s = cge.location_quotients(reg, nat, method="slq")
    assert s.loc["A", "C"] == pytest.approx(0.5) and s.loc["C", "A"] == pytest.approx(1.2)
    c = cge.location_quotients(reg, nat, method="cilq")
    assert c.loc["A", "B"] == pytest.approx(0.5 / 1.0)
    assert c.loc["C", "A"] == pytest.approx(1.2 / 0.5)
    assert c.loc["B", "B"] == pytest.approx(1.0)
    lam = math.log2(1.0 + 100.0 / 1000.0) ** 0.3
    f = cge.location_quotients(reg, nat, method="flq", delta=0.3)
    assert f.loc["A", "B"] == pytest.approx(0.5 * lam)
    assert f.loc["C", "C"] == pytest.approx(1.2 * lam)
    a = cge.location_quotients(reg, nat, method="aflq", delta=0.3)
    assert a.loc["A", "C"] == pytest.approx(0.5 / 1.2 * lam * math.log2(2.2))
    assert a.loc["A", "B"] == pytest.approx(f.loc["A", "B"])        # SLQ_B = 1: no boost
    for k in slq:
        assert s.loc[k, k] == pytest.approx(slq[k])


def test_regional_coefficients_cap_the_quotient_at_one():
    nat = pd.DataFrame([[0.2, 0.1], [0.3, 0.4]], index=["A", "B"], columns=["A", "B"])
    lq = pd.DataFrame([[0.5, 2.0], [1.5, 0.8]], index=["A", "B"], columns=["A", "B"])
    r = cge.regional_coefficients(nat, lq)
    np.testing.assert_allclose(r.to_numpy(), [[0.1, 0.1], [0.3, 0.32]])


def test_bad_quotient_inputs_are_refused():
    with pytest.raises(KeyError):
        cge.location_quotients({"A": 1.0}, {"B": 1.0})
    with pytest.raises(ValueError, match="delta"):
        cge.location_quotients({"A": 1.0}, {"A": 2.0}, delta=1.5)
    with pytest.raises(ValueError, match="method"):
        cge.location_quotients({"A": 1.0}, {"A": 2.0}, method="x")
