"""Forward-looking investment through dsgelab: the Ramsey CGE.

In a one-good closed economy with Cobb-Douglas production the Ramsey CGE is the
stochastic growth model, so it is checked against that model written directly
in dsgelab: same steady state, same first-order solution, same transition.
"""

from dataclasses import dataclass

import numpy as np
import pytest

dk = pytest.importorskip("dsgelab")

import cgelab as cge  # noqa: E402
from cgelab import bridge  # noqa: E402
from cgelab.examples import toy  # noqa: E402

A = 100.0 / (60.0 ** 0.6 * 40.0 ** 0.4)          # the one-good SAM's Cobb-Douglas scale


@dataclass
class RBC:
    alpha: float = 0.4
    beta: float = 1.0 / 1.05
    delta: float = 0.05
    rho: float = 0.9
    uc: float = 0.10


def _rbc(shock: bool):
    def eqs(n, c, e, p):
        a0 = c.a if shock else 1.0
        a1 = n.a if shock else 1.0
        y = a0 * A * 60.0 ** 0.6 * (p.uc * c.k) ** p.alpha
        y1 = a1 * A * 60.0 ** 0.6 * (p.uc * n.k) ** p.alpha
        out = [n.k - ((1 - p.delta) * c.k + y - c.c),
               c.c ** -1 - p.beta * n.c ** -1 * (p.alpha * y1 / n.k + 1 - p.delta)]
        if shock:
            out.append(np.log(n.a) - p.rho * np.log(c.a) - e.eps_a)
        return out
    states = ["k", "a"] if shock else ["k"]
    return dk.Model(states=states, jumps=["c"], shocks=["eps_a"] if shock else [],
                    equations=eqs, name="rbc")


def test_the_one_good_base_year_is_the_steady_state():
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ)
    ss = dk.steady_state(model, guess, params)
    assert ss.values["k"] == pytest.approx(400.0, rel=1e-10)
    assert ss.values["inv"] == pytest.approx(20.0, rel=1e-10)


def test_first_order_solution_is_the_growth_model():
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ, tfp_persistence=0.9)
    sol = dk.solve(model, params, guess=guess)
    ref = dk.solve(_rbc(True), RBC(), guess={"k": 400.0, "a": 1.0, "c": 80.0})
    np.testing.assert_allclose(sol.P, ref.P, atol=1e-7)
    np.testing.assert_allclose(sol.irf("eps_a", size=0.01, periods=30)["k"],
                               ref.irf("eps_a", size=0.01, periods=30)["k"], atol=1e-7)


def test_transition_is_the_growth_model():
    """The two models fix different jumps at the end of the horizon (investment
    and consumption), so a truncated path differs near T. The first forty
    periods of a long horizon are the comparison."""
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ)
    ss = dk.steady_state(model, guess, params)
    rbc = _rbc(False)
    ss2 = dk.steady_state(rbc, {"k": 400.0, "c": 80.0}, RBC())
    path = dk.perfect_foresight(model, ss, ss, params=params, T=150, initial_states=[300.0])
    ref = dk.perfect_foresight(rbc, ss2, ss2, params=RBC(), T=150, initial_states=[300.0])
    assert path.converged
    np.testing.assert_allclose(np.asarray(path["k"])[:40], np.asarray(ref["k"])[:40], atol=1e-7)


def test_an_announced_tax_moves_investment_before_it_starts():
    """run_path would not move until the tax is in force; agents who see it
    coming do."""
    _, econ = toy.build_with_institutions()
    model, params, guess = bridge.ramsey(econ)
    ss = dk.steady_state(model, guess, params)
    tax = cge.Scenario(product_tax=econ.ptax * 1.5)
    T = 80
    path_params = bridge.announced(params, [cge.Scenario()] * 5 + [tax] * (T + 1 - 5))
    terminal = dk.steady_state(model, ss.values, bridge.announced(params, [tax])[0])
    path = dk.perfect_foresight(model, ss, terminal, params=path_params, T=T)
    assert path.converged
    inv = np.asarray(path["inv"])
    assert abs(inv[0] - ss.values["inv"]) > 1e-3 * ss.values["inv"]


def test_the_period_equilibrium_is_cgelabs():
    _, econ = toy.build_one_good()
    res = bridge.period_at(econ, cge.Closure(investment="fixed_real"), cge.Scenario(), 400.0, 20.0,
                           user_cost=0.10)
    assert res.converged and abs(res.walras) < 1e-10
    assert res.factor_prices[econ.factor("f_cap")] == pytest.approx(1.0, rel=1e-10)
