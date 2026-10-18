"""Investment buys a chosen commodity basket. The capital-stock allocation is unchanged."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy

CLOSURES = [
    cge.Closure(),
    cge.Closure(foreign="fixed_rate"),
    cge.Closure(investment="fixed_real"),
    cge.Closure(government="nominal_fixed"),
    cge.Closure(investment="fixed_real", foreign="fixed_rate"),
]


def test_an_invalid_mix_is_named_and_refused():
    _, econ = toy.build()
    with pytest.raises(ValueError, match="investment_mix"):
        cge.solve_period(econ, scenario=cge.Scenario(investment_mix=np.array([0.2, 0.2])))


def test_the_base_mix_is_the_sam_composition():
    _, econ = toy.build()
    period = cge.solve_period(econ)
    base_q = econ.inv_dem / (1.0 + econ.ptax)
    assert np.allclose(period.investment_quantity, base_q, atol=1e-8)


@pytest.mark.parametrize("closure", CLOSURES, ids=lambda c: c.describe())
def test_a_mix_shifts_output_and_keeps_walras(closure):
    _, econ = toy.build()
    mix = np.zeros(econ.n_com)
    mix[econ.commodity("c_agr")] = 1.0
    base = cge.solve_period(econ, closure)
    shock = cge.solve_period(econ, closure, cge.Scenario(investment_mix=mix))
    assert shock.converged
    assert abs(shock.walras) < 1e-8
    agr = econ.commodity("c_agr")
    assert shock.Q[agr] > base.Q[agr]
    assert shock.investment_quantity[agr] > base.investment_quantity[agr]


def test_to_jump_reads_the_mix(tmp_path):
    _, econ = toy.build()
    mix = np.array([0.8, 0.2])
    dados = cge.to_jump(econ, scenario=cge.Scenario(investment_mix=mix), path=tmp_path / "m.jl")
    assert np.allclose(dados["inv_w"], mix)
