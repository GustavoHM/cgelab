"""The Armington nest at an elasticity of one, its Cobb-Douglas limit."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.functions import armington, cobb_douglas_price

SD = np.array([0.7, 0.4, 1.0, 0.0])
PD = np.array([0.9, 1.3, 1.1, 1.0])
PM = np.array([0.8, 1.1, 1.0, 1.2])


def test_unit_elasticity_is_the_cobb_douglas_limit():
    """At sigma = 1 the composite price used to be 1 whatever the prices."""
    pq, dsh, msh = armington(SD, PD, PM, 1.0)
    shares = np.vstack([SD, 1.0 - SD])
    prices = np.vstack([PD, PM])
    expected = np.array([cobb_douglas_price(shares[:, [j]], prices[:, j])[0] for j in range(len(SD))])
    assert np.allclose(pq, expected, rtol=1e-12)
    assert np.allclose(dsh, np.where(PD > 0, SD * pq / PD, 0.0))
    assert np.allclose(msh, (1.0 - SD) * pq / PM)
    assert np.allclose(pq, PD * dsh + PM * msh)          # value is conserved


@pytest.mark.parametrize("eps", [1e-5, -1e-5])
def test_unit_elasticity_is_continuous(eps):
    at_one, near = armington(SD, PD, PM, 1.0), armington(SD, PD, PM, 1.0 + eps)
    for a, b in zip(at_one, near):
        assert np.allclose(a, b, rtol=1e-4)


def test_mixed_elasticities_by_commodity():
    sigma = np.array([1.0, 2.0, 1.0, 0.5])
    pq, _, _ = armington(SD, PD, PM, sigma)
    for j, s in enumerate(sigma):
        assert np.isclose(pq[j], armington(SD[[j]], PD[[j]], PM[[j]], s)[0][0])


def test_a_world_price_shock_moves_a_cobb_douglas_composite():
    """End to end: with sigma_armington = 1 a dearer import raises the consumer price."""
    _, econ = toy.build(toy.spec(sigma_armington=1.0, export_rule="cet"))
    base = cge.solve_period(econ)
    shocked = cge.solve_period(econ, scenario=cge.Scenario(world_price=np.array([1.5, 1.0])))
    assert shocked.pq[0] / base.pq[0] > 1.01
