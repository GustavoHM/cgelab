"""Destination-specific origin nests and bilateral domestic sales."""

import numpy as np

import cgelab as cge
from cgelab.examples import regions


def test_base_year_bilateral_cells_come_from_the_sam():
    _, econ = regions.build()
    result = cge.solve_period(econ)
    expected = np.array([[[45.0], [18.0]], [[18.0], [45.0]]])
    expected_sam = np.array([[[50.0], [20.0]], [[20.0], [50.0]]])
    np.testing.assert_allclose(result.bilateral, expected, atol=1e-8)
    np.testing.assert_allclose(result.bilateral + result.bilateral_imports,
                               expected_sam, atol=1e-8)
    national = np.array([27.0, 27.0])
    np.testing.assert_allclose(
        result.bilateral.sum(axis=1)[:, 0] + national,
        result.Q - result.exports, atol=1e-8)


def test_origin_prices_change_destination_demand_not_only_allocation():
    _, econ = regions.build()
    base = cge.solve_period(econ)
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("r1:a")] = 1.2
    result = cge.solve_period(econ, scenario=cge.Scenario(productivity=tfp),
                              warm_start=base)
    assert result.converged and abs(result.walras) < 1e-8
    assert result.bilateral[0, 0, 0] > base.bilateral[0, 0, 0]
    assert result.bilateral[0, 1, 0] > base.bilateral[0, 1, 0]
    assert not np.allclose(result.bilateral / base.bilateral, 1.0)


def test_bilateral_model_is_homogeneous_of_degree_zero():
    _, econ = regions.build()
    base = cge.solve_period(econ)
    scaled = cge.solve_period(econ, cge.Closure(numeraire=2.0))
    np.testing.assert_allclose(scaled.X, base.X, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(scaled.bilateral, base.bilateral, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(scaled.pq, 2.0 * base.pq, rtol=1e-8, atol=1e-8)
    assert abs(scaled.walras) < 1e-8


def test_cet_and_bilateral_destinations_compose():
    _, econ = regions.build(regions.spec(export_rule="cet", sigma_cet=3.0))
    base = cge.solve_period(econ)
    result = cge.solve_period(econ, scenario=cge.Scenario(export_price=[1.1]),
                              warm_start=base)
    assert result.converged and abs(result.walras) < 1e-8
    assert (result.exports > base.exports).all()
    assert result.bilateral.shape == (2, 2, 1)
