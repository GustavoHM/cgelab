"""Subtotals (Harrison, Horridge and Pearson, 2000): closed forms and the model."""

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import toy

X0 = {"a": np.array(1.0), "b": np.array(2.0), "c": np.array(0.5)}
X1 = {"a": np.array(3.0), "b": np.array(5.0), "c": np.array(1.5)}


def test_contributions_of_a_product_are_the_line_integrals():
    """y = a b + c^2 along the straight line: a gets da (b0 + db/2), b gets db (a0 + da/2)."""
    d = ck.path_decomposition(lambda x: float(x["a"] * x["b"] + x["c"] ** 2), X0, X1,
                              {"A": ["a"], "B": ["b"], "C": ["c"]})
    row = d.loc["result"]
    assert row.A == pytest.approx(2.0 * (2.0 + 1.5), rel=1e-9)
    assert row.B == pytest.approx(3.0 * (1.0 + 1.0), rel=1e-9)
    assert row.C == pytest.approx(1.5 ** 2 - 0.5 ** 2, rel=1e-9)
    assert abs(row.residual) < 1e-9


def test_an_additive_result_gives_each_shock_its_own_effect():
    f = lambda x: float(np.exp(x["a"]) + np.sin(x["b"]) + x["c"] ** 3)  # noqa: E731
    d = ck.path_decomposition(f, X0, X1, {"A": ["a"], "B": ["b"], "C": ["c"]}).loc["result"]
    # Central differences with step 1e-4 carry an error of order 1e-8 here.
    assert d.A == pytest.approx(np.exp(3.0) - np.exp(1.0), rel=1e-7)
    assert d.B == pytest.approx(np.sin(5.0) - np.sin(2.0), rel=1e-7)
    assert d.C == pytest.approx(1.5 ** 3 - 0.5 ** 3, rel=1e-7)


def test_groups_must_cover_every_shock_once():
    f = lambda x: float(x["a"])  # noqa: E731
    with pytest.raises(ValueError, match="in no group"):
        ck.path_decomposition(f, X0, X1, {"A": ["a"], "B": ["b"]})
    with pytest.raises(ValueError, match="groups"):
        ck.path_decomposition(f, X0, X1, {"A": ["a", "b"], "B": ["b", "c"]})


def _measure(p):
    return {"gdp": ck.real_gdp(p)["expenditure"], "er": p.exchange_rate,
            "imports": float(p.imports.sum())}


def test_subtotals_add_up_to_the_solved_change():
    _, econ = toy.build_with_trade_taxes()
    alt = ck.Scenario(tariff=np.zeros(2), world_price=np.array([1.1, 0.95]),
                      product_tax=econ.ptax * 1.2)
    d = ck.subtotals(econ, ck.Scenario(), alt,
                     {"tariff": ["tariff"], "world": ["world_price"], "tax": ["product_tax"]},
                     measure=_measure, nodes=4)
    assert (d.residual.abs() < 1e-8 * np.maximum(d.change.abs(), 1)).all()
    direct = _measure(ck.solve_period(econ, scenario=alt))
    for k, v in direct.items():
        assert d.loc[k, "reference"] + d.loc[k, "change"] == pytest.approx(v, rel=1e-12)


def test_one_group_is_the_whole_change_and_a_still_group_is_zero():
    _, econ = toy.build_with_trade_taxes()
    alt = ck.Scenario(tariff=np.zeros(2))
    d = ck.subtotals(econ, ck.Scenario(), alt, {"tariff": ["tariff"], "tax": ["product_tax"]},
                     measure=_measure, nodes=4)
    assert (d.tax == 0).all()
    np.testing.assert_allclose(d.tariff, d.change, rtol=1e-8)


def test_mapping_fields_and_closures():
    _, econ = toy.build_with_own_production()
    alt = ck.Scenario(activity_scale={"a_own": 1.5}, world_price=np.array([1.2, 1.0]))
    d = ck.subtotals(econ, ck.Scenario(), alt,
                     {"own": ["activity_scale"], "world": ["world_price"]},
                     measure=_measure, closure=ck.Closure(foreign="fixed_rate"), nodes=4)
    assert (d.residual.abs() < 1e-8 * np.maximum(d.change.abs(), 1)).all()
    # Both shocks move GDP; with capital in fixed supply, more home production
    # draws it out of the market activities, so the sign is not obvious.
    assert abs(d.loc["gdp", "own"]) > 1e-3 and abs(d.loc["gdp", "world"]) > 1e-3


def test_rules_that_differ_have_no_path_between_them():
    _, econ = toy.build()
    with pytest.raises(ValueError, match="not numbers"):
        ck.subtotals(econ, ck.Scenario(wedge_revenue_to="capital"),
                     ck.Scenario(wedge_revenue_to="lumpsum"), {}, measure=_measure)
