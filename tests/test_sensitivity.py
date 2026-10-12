"""Systematic sensitivity: Stroud quadrature against closed forms, and on the model."""

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import toy

PARAMS = {"x1": ck.Uniform(1.0, 3.0), "x2": ck.Normal(0.5, 0.2), "x3": ck.Uniform(-1.0, 2.0)}


def cubic(v):
    x1, x2, x3 = v["x1"], v["x2"], v["x3"]
    return 2.0 + 3.0 * x1 - x2 * x3 + 0.5 * x1 ** 3 + 4.0 * x2 ** 2 - x1 * x2 * x3


def cubic_mean():
    """E[cubic] in closed form: independent parameters, symmetric distributions."""
    m = {k: d.mean for k, d in PARAMS.items()}
    s2 = {k: d.sd ** 2 for k, d in PARAMS.items()}
    e_x1_cubed = m["x1"] ** 3 + 3 * m["x1"] * s2["x1"]
    return (2.0 + 3.0 * m["x1"] - m["x2"] * m["x3"] + 0.5 * e_x1_cubed
            + 4.0 * (m["x2"] ** 2 + s2["x2"]) - m["x1"] * m["x2"] * m["x3"])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_stroud_points_reproduce_the_first_three_moments(n):
    z = ck.stroud_points(n)
    assert z.shape == (2 * n, n)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose((z ** 2).mean(axis=0), 1.0, atol=1e-14)
    np.testing.assert_allclose((z ** 3).mean(axis=0), 0.0, atol=1e-14)
    cross = (z.T @ z) / len(z)
    np.testing.assert_allclose(cross, np.eye(n), atol=1e-14)
    if n >= 3:
        triple = (z[:, 0] * z[:, 1] * z[:, 2]).mean()
        assert abs(triple) < 1e-14


def test_quadrature_mean_is_exact_for_a_cubic():
    res = ck.sensitivity(cubic, PARAMS)
    assert len(res.points) == 6
    assert res.summary().loc["result", "mean"] == pytest.approx(cubic_mean(), rel=1e-13)


def test_quadrature_sd_is_exact_for_a_linear_result():
    res = ck.sensitivity(lambda v: {"y": 1 + 2 * v["x1"] - 3 * v["x2"] + v["x3"]}, PARAMS)
    sd = np.sqrt(4 * PARAMS["x1"].sd ** 2 + 9 * PARAMS["x2"].sd ** 2 + PARAMS["x3"].sd ** 2)
    assert res.summary().loc["y", "sd"] == pytest.approx(sd, rel=1e-13)


def test_quadrature_points_stay_inside_a_uniform_support():
    res = ck.sensitivity(lambda v: v["x1"] + v["x3"], PARAMS)
    assert res.points.x1.between(1.0, 3.0).all() and res.points.x3.between(-1.0, 2.0).all()


def test_monte_carlo_agrees_and_is_reproducible():
    a = ck.sensitivity(cubic, PARAMS, method="monte_carlo", draws=20000, seed=1)
    b = ck.sensitivity(cubic, PARAMS, method="monte_carlo", draws=20000, seed=1)
    pd_equal = a.points.equals(b.points)
    assert pd_equal
    s = a.summary()
    assert s.loc["result", "mean"] == pytest.approx(cubic_mean(), rel=5e-3)
    assert {"q0.05", "q0.5", "q0.95", "min", "max"} <= set(s.columns)


def test_asymmetric_distributions_need_monte_carlo():
    params = {"x": ck.Triangular(0.0, 0.2, 1.0)}
    with pytest.raises(ValueError, match="symmetric"):
        ck.sensitivity(lambda v: v["x"], params)
    res = ck.sensitivity(lambda v: v["x"], params, method="monte_carlo", draws=50000)
    assert res.summary().loc["result", "mean"] == pytest.approx(0.4, rel=1e-2)
    assert ck.Triangular(0.0, 0.5, 1.0).symmetric


def _real_gdp_change(base, alt):
    return {"gdp_pct": 100 * (ck.real_gdp(alt)["expenditure"]
                              / ck.real_gdp(base)["expenditure"] - 1),
            "er_pct": 100 * (alt.exchange_rate / base.exchange_rate - 1)}


def test_spec_sweep_over_elasticities():
    sam, econ = toy.build()
    run = ck.spec_sweep(sam, toy.spec(), measure=_real_gdp_change,
                        scenario=ck.Scenario(product_tax=econ.ptax * 1.5))
    params = {"sigma_va": ck.Uniform(0.3, 1.3), "sigma_armington[c_ind]": ck.Uniform(1.0, 3.0)}
    quad = ck.sensitivity(run, params)
    mc = ck.sensitivity(run, params, method="monte_carlo", draws=60, seed=3)
    q, m = quad.summary(), mc.summary()
    assert q.loc["gdp_pct", "mean"] == pytest.approx(m.loc["gdp_pct", "mean"], abs=3 * m.loc[
        "gdp_pct", "sd"] / np.sqrt(60) + 1e-6)
    # A point at the central values reproduces a direct solve.
    direct = run({"sigma_va": 0.8, "sigma_armington[c_ind]": 2.0})
    spec = toy.spec(sigma_va=0.8, sigma_armington={"__default__": 2.0, "c_ind": 2.0})
    e = ck.calibrate(sam, spec)
    alt = ck.solve_period(e, scenario=ck.Scenario(product_tax=e.ptax * 1.5))
    assert direct["gdp_pct"] == pytest.approx(
        _real_gdp_change(ck.solve_period(e), alt)["gdp_pct"], rel=1e-10)


def test_spec_sweep_refuses_an_unknown_field():
    sam, _ = toy.build()
    run = ck.spec_sweep(sam, toy.spec(), measure=lambda b, a: 0.0)
    with pytest.raises(KeyError, match="sigma_foo"):
        run({"sigma_foo": 1.0})
