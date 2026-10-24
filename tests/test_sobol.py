"""Sobol indices on Saltelli's design.

The indices are checked on functions whose indices are known in closed form:
Ishigami's, a linear model, an additive one (no interactions, so the total
index is the first-order one) and a pure interaction (no first-order effect at
all). SALib, given the same design, must give the same numbers; that check runs
when SALib is installed, and ``validation/sobol_salib.py`` runs it on the toy
economy too.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import cgelab as cge
from cgelab.examples import toy

PI = np.pi
ISHIGAMI = {f"x{i}": cge.Uniform(-PI, PI) for i in (1, 2, 3)}


def ishigami(v, a=7.0, b=0.1):
    return np.sin(v["x1"]) + a * np.sin(v["x2"]) ** 2 + b * v["x3"] ** 4 * np.sin(v["x1"])


def ishigami_indices(a=7.0, b=0.1):
    v1 = 0.5 * (1 + b * PI ** 4 / 5) ** 2
    v2 = a ** 2 / 8
    v13 = b ** 2 * PI ** 8 / 18 - b ** 2 * PI ** 8 / 50
    v = v1 + v2 + v13
    return np.array([v1, v2, 0.0]) / v, np.array([v1 + v13, v2, v13]) / v


def test_the_quantile_functions_are_scipys():
    u = np.linspace(0.001, 0.999, 101)
    np.testing.assert_allclose(cge.Uniform(2, 5).ppf(u), stats.uniform(2, 3).ppf(u), rtol=1e-14)
    np.testing.assert_allclose(cge.Normal(1, 0.3).ppf(u), stats.norm(1, 0.3).ppf(u), rtol=1e-14)
    tri = cge.Triangular(1.0, 1.5, 4.0)
    np.testing.assert_allclose(tri.ppf(u), stats.triang((1.5 - 1) / 3, 1.0, 3.0).ppf(u),
                               rtol=1e-12)


def test_ishigami_in_closed_form():
    res = cge.sobol(ishigami, ISHIGAMI, n=8192, bootstrap=100)
    idx = res.indices().loc["result"]
    s1, st = ishigami_indices()
    np.testing.assert_allclose(idx["S1"], s1, atol=0.01)
    np.testing.assert_allclose(idx["ST"], st, atol=0.01)
    # The intervals cover the truth, and the interaction of x1 and x3 shows as ST - S1.
    assert (np.abs(idx["S1"] - s1) < idx["S1_conf"]).all()
    assert (np.abs(idx["ST"] - st) < idx["ST_conf"]).all()
    assert idx.loc["x3", "ST"] - idx.loc["x3", "S1"] > 0.2


def test_a_linear_model_and_an_additive_one():
    params = {"a": cge.Normal(0.0, 1.0), "b": cge.Normal(2.0, 2.0), "c": cge.Uniform(0.0, 3.0)}
    coef = {"a": 1.0, "b": 0.5, "c": 2.0}
    res = cge.sobol(lambda v: sum(coef[k] * v[k] for k in coef), params, n=4096, bootstrap=0)
    var = np.array([1.0 * 1.0, 0.25 * 4.0, 4.0 * 9.0 / 12.0])
    idx = res.indices().loc["result"]
    np.testing.assert_allclose(idx["S1"], var / var.sum(), atol=0.01)
    np.testing.assert_allclose(idx["ST"], var / var.sum(), atol=0.01)
    assert np.isnan(idx["S1_conf"]).all()          # bootstrap=0: no intervals
    # Additive and nonlinear: still no interaction, so ST equals S1.
    bounded = {k: cge.Uniform(0.0, 1.0) for k in ("a", "b", "c")}
    add = cge.sobol(lambda v: np.exp(v["a"]) + v["b"] ** 3, bounded, n=4096, bootstrap=0)
    i = add.indices().loc["result"]
    np.testing.assert_allclose(i["ST"], i["S1"], atol=0.02)
    assert i.loc["c", "ST"] == pytest.approx(0.0, abs=1e-12)


def test_a_pure_interaction_has_no_first_order_effect():
    params = {"x": cge.Uniform(-1, 1), "y": cge.Uniform(-1, 1)}
    res = cge.sobol(lambda v: v["x"] * v["y"], params, n=4096, bootstrap=0)
    idx = res.indices().loc["result"]
    np.testing.assert_allclose(idx["S1"], 0.0, atol=0.02)
    np.testing.assert_allclose(idx["ST"], 1.0, atol=0.03)


def test_the_design_and_the_draws():
    res = cge.sobol(ishigami, ISHIGAMI, n=64, bootstrap=0)
    p = res.points
    assert len(p) == 64 * (3 + 2)
    assert p["matrix"].value_counts().to_dict() == {"A": 64, "B": 64, "AB_x1": 64,
                                                    "AB_x2": 64, "AB_x3": 64}
    assert p["weight"].sum() == pytest.approx(1.0)
    # AB_i is A with column i from B.
    A = p[p["matrix"] == "A"].set_index("row")
    B = p[p["matrix"] == "B"].set_index("row")
    AB2 = p[p["matrix"] == "AB_x2"].set_index("row")
    np.testing.assert_array_equal(AB2["x2"], B["x2"])
    np.testing.assert_array_equal(AB2[["x1", "x3"]], A[["x1", "x3"]])
    w = p["weight"]
    assert res.summary().loc["result", "mean"] == pytest.approx(float((w * p["result"]).sum()))
    # The same seed gives the same design; another seed another.
    again = cge.sobol(ishigami, ISHIGAMI, n=64, bootstrap=0)
    pd.testing.assert_frame_equal(again.points, res.points)
    other = cge.sobol(ishigami, ISHIGAMI, n=64, bootstrap=0, seed=7)
    assert not np.allclose(other.points["x1"], res.points["x1"])


def test_against_salib_on_the_same_design():
    analyze = pytest.importorskip("SALib.analyze.sobol")
    sample = pytest.importorskip("SALib.sample.sobol")
    res = cge.sobol(ishigami, ISHIGAMI, n=1024, bootstrap=0)
    problem = {"num_vars": 3, "names": list(ISHIGAMI), "bounds": [[-PI, PI]] * 3}
    X = sample.sample(problem, 1024, calc_second_order=False, scramble=True, seed=res.seed)
    A, B, AB = res._arrays("result")
    order = np.column_stack([A, AB, B]).ravel()
    np.testing.assert_allclose(X[::5], res.points.loc[res.points["matrix"] == "A",
                                                      list(ISHIGAMI)].to_numpy(), rtol=1e-15)
    Si = analyze.analyze(problem, order, calc_second_order=False, print_to_console=False,
                         num_resamples=2, seed=1)
    idx = res.indices().loc["result"]
    np.testing.assert_allclose(idx["S1"], Si["S1"], atol=1e-13)
    np.testing.assert_allclose(idx["ST"], Si["ST"], atol=1e-13)


def test_the_toy_economy():
    sam, econ = toy.build()
    run = cge.spec_sweep(sam, toy.spec(), scenario=cge.Scenario(product_tax=econ.ptax * 1.5),
                         measure=lambda b, a: {"gdp_pct": 100 * (
                             cge.real_gdp(a)["expenditure"] / cge.real_gdp(b)["expenditure"] - 1),
                             "er": a.exchange_rate})
    params = {"sigma_va": cge.Uniform(0.3, 1.3), "sigma_armington": cge.Uniform(1.5, 3.5)}
    res = cge.sobol(run, params, n=64, bootstrap=50)
    idx = res.indices()
    assert set(idx.index.get_level_values("output")) == {"gdp_pct", "er"}
    assert np.isfinite(idx[["S1", "ST"]].to_numpy()).all()
    assert (idx["ST"] >= idx["S1"] - idx["S1_conf"] - idx["ST_conf"]).all()
    # The Armington elasticity carries real GDP's variance, as the quadrature suggests.
    assert idx.loc[("gdp_pct", "sigma_armington"), "ST"] > idx.loc[("gdp_pct", "sigma_va"), "ST"]


def test_a_constant_result_and_what_sobol_refuses():
    params = {"x": cge.Uniform(0, 1)}
    res = cge.sobol(lambda v: 3.0, params, n=16, bootstrap=0)
    assert res.indices()[["S1", "ST"]].isna().all().all()
    with pytest.raises(ValueError, match="power of two"):
        cge.sobol(ishigami, ISHIGAMI, n=1000)
    with pytest.raises(ValueError, match="no parameters"):
        cge.sobol(ishigami, {})
    with pytest.raises(ValueError, match="conf_level"):
        cge.sobol(ishigami, ISHIGAMI, n=16, conf_level=1.0)
    with pytest.raises(ValueError, match="bootstrap"):
        cge.sobol(ishigami, ISHIGAMI, n=16, bootstrap=1)

    class NoQuantile:
        mean, sd, symmetric = 0.0, 1.0, True

    with pytest.raises(TypeError, match="ppf"):
        cge.sobol(lambda v: v["z"], {"z": NoQuantile()}, n=16)
