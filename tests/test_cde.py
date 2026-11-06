"""CDE household demand: closed-form elasticities, the CES case, and the base year."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.functions import cde_expenditure, cde_shares, cde_utility

SAM = toy._frame(toy._BASE)
B_SUB = {"c_agr": 0.3, "c_ind": 0.6}
ETA = {"c_agr": 0.6, "c_ind": 1.2}


def _econ(**kw):
    return cge.calibrate(SAM, toy.spec(household_demand="cde", cde_substitution=B_SUB, **kw),
                         income_elasticities=ETA)


ECON = _econ()


def _demand(h, p, E):
    c = ECON.cde
    return cde_shares(c["B"][h], c["b"], c["e"][h], p, E) * E / p


def test_the_base_year_reproduces():
    period = cge.solve_period(ECON)
    assert period.converged and abs(period.walras) < 1e-12
    value = period.hh_quantity * (1.0 + ECON.ptax)[None, :]
    np.testing.assert_allclose(value, ECON.hh_budget * ECON.hh_spend0[:, None], rtol=1e-12)
    np.testing.assert_allclose(period.real["hh_price_index"], 1.0, atol=1e-12)


@pytest.mark.parametrize("h", [0, 1])
def test_income_elasticities_are_the_ones_calibrated(h):
    """eta_i = 1 - b_i + sum w b + (e_i b_i - sum w e b) / sum w e, scaled to Engel."""
    c = ECON.cde
    p0 = 1.0 + ECON.ptax
    E0 = float(ECON.hh_spend0[h])
    step = 1e-6
    q1, q0 = _demand(h, p0, E0 * (1 + step)), _demand(h, p0, E0 * (1 - step))
    numeric = (np.log(q1) - np.log(q0)) / (np.log(1 + step) - np.log(1 - step))
    w, b, e = ECON.hh_budget[h], c["b"], c["e"][h]
    formula = 1 - b + w @ b + (e * b - w @ (e * b)) / (w @ e)
    np.testing.assert_allclose(numeric, formula, rtol=1e-7)
    eta = np.array([ETA["c_agr"], ETA["c_ind"]])
    np.testing.assert_allclose(numeric, eta / (w @ eta), rtol=1e-7)


@pytest.mark.parametrize("h", [0, 1])
def test_price_elasticities_match_the_closed_form(h):
    """d ln w_i / d ln p_j = b_i delta_ij - b_j w_j - w_j (e_i b_i - sum w e b) / sum w e."""
    c = ECON.cde
    p0 = 1.0 + ECON.ptax
    E0 = float(ECON.hh_spend0[h])
    w, b, e = ECON.hh_budget[h], c["b"], c["e"][h]
    step = 1e-6
    for j in range(2):
        up, dn = p0.copy(), p0.copy()
        up[j] *= 1 + step
        dn[j] *= 1 - step
        numeric = ((np.log(_demand(h, up, E0)) - np.log(_demand(h, dn, E0)))
                   / (np.log(1 + step) - np.log(1 - step)))
        dlnw = (b * (np.arange(2) == j) - b[j] * w[j]
                - w[j] * (e * b - w @ (e * b)) / (w @ e))
        np.testing.assert_allclose(numeric, dlnw - (np.arange(2) == j), rtol=1e-6, atol=1e-9)


def test_equal_substitution_and_unit_income_elasticities_are_a_ces():
    econ = cge.calibrate(SAM, toy.spec(household_demand="cde", cde_substitution=0.4))
    c = econ.cde
    p0 = 1.0 + econ.ptax
    p1 = p0 * np.array([1.3, 0.9])
    sigma = 1.0 - 0.4
    for h in range(econ.n_hh):
        w0 = econ.hh_budget[h]
        ces = w0 * (p1 / p0) ** (1 - sigma)
        ces = ces / ces.sum()
        got = cde_shares(c["B"][h], c["b"], c["e"][h], p1, 2.0 * econ.hh_spend0[h])
        np.testing.assert_allclose(got, ces, rtol=1e-12)


def test_utility_and_expenditure_are_inverses():
    c = ECON.cde
    p = (1.0 + ECON.ptax) * np.array([1.2, 0.8])
    u = cde_utility(c["B"][0], c["b"], c["e"][0], p, 90.0)
    assert cde_expenditure(c["B"][0], c["b"], c["e"][0], p, u) == pytest.approx(90.0, rel=1e-12)


def test_a_shock_solves_and_the_money_metric_index_falls_when_goods_get_cheaper():
    r = cge.solve_period(ECON, scenario=cge.Scenario(productivity=np.array([1.1, 1.0])))
    assert r.converged and abs(r.walras) < 1e-12
    assert np.all(r.real["hh_price_index"] < 1.0)


def test_cde_needs_its_parameter_and_refuses_wedges():
    with pytest.raises(ValueError, match="cde_substitution"):
        cge.calibrate(SAM, toy.spec(household_demand="cde"))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        cge.calibrate(SAM, toy.spec(household_demand="cde", cde_substitution=1.2))
    with pytest.raises(ValueError, match="affine"):
        cge.solve_period(ECON, scenario=cge.Scenario(
            consumer_wedge=cge.consumer_wedge(ECON, {"c_ind": 0.1})))
