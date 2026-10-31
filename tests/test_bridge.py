"""Forward-looking investment through dsgelab: the Ramsey CGE.

In a one-good closed economy with Cobb-Douglas production the Ramsey CGE is the
stochastic growth model, so it is checked against that model written directly
in dsgelab: same steady state, same first-order solution, same transition.
"""

from dataclasses import dataclass

import numpy as np
import pytest

dsg = pytest.importorskip("dsgelab")

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
    return dsg.Model(states=states, jumps=["c"], shocks=["eps_a"] if shock else [],
                     equations=eqs, name="rbc")


def test_the_one_good_base_year_is_the_steady_state():
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ)
    ss = dsg.steady_state(model, guess, params)
    assert ss.values["k"] == pytest.approx(400.0, rel=1e-10)
    assert ss.values["inv"] == pytest.approx(20.0, rel=1e-10)


def test_first_order_solution_is_the_growth_model():
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ, tfp_persistence=0.9)
    sol = dsg.solve(model, params, guess=guess)
    ref = dsg.solve(_rbc(True), RBC(), guess={"k": 400.0, "a": 1.0, "c": 80.0})
    np.testing.assert_allclose(sol.P, ref.P, atol=1e-7)
    np.testing.assert_allclose(sol.irf("eps_a", size=0.01, periods=30)["k"],
                               ref.irf("eps_a", size=0.01, periods=30)["k"], atol=1e-7)


def test_transition_is_the_growth_model():
    """The two models fix different jumps at the end of the horizon (investment
    and consumption), so a truncated path differs near T. The first forty
    periods of a long horizon are the comparison."""
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ)
    ss = dsg.steady_state(model, guess, params)
    rbc = _rbc(False)
    ss2 = dsg.steady_state(rbc, {"k": 400.0, "c": 80.0}, RBC())
    path = dsg.perfect_foresight(model, ss, ss, params=params, T=150, initial_states=[300.0])
    ref = dsg.perfect_foresight(rbc, ss2, ss2, params=RBC(), T=150, initial_states=[300.0])
    assert path.converged
    np.testing.assert_allclose(np.asarray(path["k"])[:40], np.asarray(ref["k"])[:40], atol=1e-7)


def test_an_announced_tax_moves_investment_before_it_starts():
    """run_path would not move until the tax is in force; agents who see it
    coming do."""
    _, econ = toy.build_with_institutions()
    model, params, guess = bridge.ramsey(econ)
    ss = dsg.steady_state(model, guess, params)
    tax = cge.Scenario(product_tax=econ.ptax * 1.5)
    T = 80
    path_params = bridge.announced(params, [cge.Scenario()] * 5 + [tax] * (T + 1 - 5))
    terminal = dsg.steady_state(model, ss.values, bridge.announced(params, [tax])[0])
    path = dsg.perfect_foresight(model, ss, terminal, params=path_params, T=T)
    assert path.converged
    inv = np.asarray(path["inv"])
    assert abs(inv[0] - ss.values["inv"]) > 1e-3 * ss.values["inv"]


def test_the_period_equilibrium_is_cgelabs():
    _, econ = toy.build_one_good()
    res = bridge.period_at(econ, cge.Closure(investment="fixed_real"), cge.Scenario(), 400.0, 20.0,
                           user_cost=0.10)
    assert res.converged and abs(res.walras) < 1e-10
    assert res.factor_prices[econ.factor("f_cap")] == pytest.approx(1.0, rel=1e-10)


def test_several_stocks_are_states_of_their_own():
    """Two regional stocks bought with one bundle: in the steady state each
    rental price equals the price of investment."""
    from cgelab.examples import regions
    _, econ = regions.build()
    factors = ["r1:f_cap", "r2:f_cap"]
    model, params, guess = bridge.ramsey(econ, capital_factor=factors)
    assert set(guess) == {"k_r1_f_cap", "k_r2_f_cap", "inv_r1_f_cap", "inv_r2_f_cap"}
    ss = dsg.steady_state(model, guess, params)
    v = ss.values
    res = bridge.period_at(econ, cge.Closure(investment="fixed_real"), cge.Scenario(),
                           [v["k_r1_f_cap"], v["k_r2_f_cap"]],
                           v["inv_r1_f_cap"] + v["inv_r2_f_cap"],
                           capital_factor=factors, user_cost=params.user_cost)
    p_i = res.savings / res.real["inv_real"]
    rents = res.factor_prices[[econ.factor(f) for f in factors]]
    np.testing.assert_allclose(rents, p_i, rtol=1e-8)


def test_one_stock_in_a_list_is_the_one_stock_model():
    _, econ = toy.build_one_good()
    model, params, guess = bridge.ramsey(econ, capital_factor=["f_cap"])
    ss = dsg.steady_state(model, guess, params)
    assert ss.values["k_f_cap"] == pytest.approx(400.0, rel=1e-10)
    assert ss.values["inv_f_cap"] == pytest.approx(20.0, rel=1e-10)


def test_a_stock_short_of_its_steady_state_draws_the_investment():
    _, econ = toy.build_with_specific_capital()
    model, params, guess = bridge.ramsey(econ, capital_factor=["f_cap", "f_kind"],
                                         depreciation={"f_cap": 0.05, "f_kind": 0.08})
    assert params.depreciation == (0.05, 0.08)
    ss = dsg.steady_state(model, guess, params)
    path = dsg.perfect_foresight(model, ss, ss, params=params, T=40,
                                 initial_states=[ss.values["k_f_cap"],
                                                 0.7 * ss.values["k_f_kind"]])
    assert path.converged
    assert np.asarray(path["inv_f_kind"])[0] > ss.values["inv_f_kind"]
    assert np.asarray(path["inv_f_cap"])[0] < ss.values["inv_f_cap"]
    with pytest.raises(KeyError, match="f_kind"):
        bridge.ramsey(econ, capital_factor=["f_cap", "f_kind"], depreciation={"f_cap": 0.05})


def test_a_failed_period_says_where_on_the_path():
    _, econ = toy.build_with_specific_capital()
    model, params, _ = bridge.ramsey(econ, capital_factor=["f_cap", "f_kind"])
    dated = bridge.announced(params, [cge.Scenario()] * 3)
    assert [p.date for p in dated] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="every start point failed") as info:
        model.residuals(np.ones(4), np.array([-50.0, 10.0, 1.0, 1.0]), np.zeros(0),
                        dated[2], None)
    note = "\n".join(info.value.__notes__)
    assert "date 2, the current period" in note
    assert "k_f_cap=-50" in note and "k_f_kind=10" in note
    assert "'baseline' is the base year" in note
    assert "negative budget for commodities" in str(info.value)
