"""Forward-looking investment: a Ramsey CGE, with dsgelab solving the intertemporal part.

``run_path`` solves one within-period equilibrium after another, and no agent
looks ahead. This module builds the other object: a representative household
that chooses investment by an Euler equation, while every period's
equilibrium is still the SAM model solved here. The two packages split the
work along that line. dsgelab never reads a SAM; it sees one state (the capital
stock), one jump (real investment) and the residuals below, and it brings the
steady state, the perfect-foresight transition, first- and second-order
solutions and impulse responses. A SAM of a few hundred accounts does not
become a DSGE with a few hundred variables: the static block is substituted
out by :func:`~cgelab.solve.solve_period`.

The period equilibrium, given the capital stock ``k`` and real investment
``inv``:

* capital services are ``uc0 * k``, with ``uc0 = discount + depreciation`` the
  base-year user cost, which is how :func:`~cgelab.dynamics.run_path` turns a
  stock into the service flow the SAM records;
* investment is fixed in real terms at ``inv`` (base-year purchaser prices),
  financed as ``Closure(investment="fixed_real")`` finances it;
* the household's consumption ``C`` is real household consumption at base-year
  prices, ``P_C`` its price, ``P_I`` the price of the investment bundle and
  ``r`` the rental price of capital services.

The intertemporal residuals:

* ``k' = (1 - delta) k + inv``;
* ``C^-g P_I / P_C = beta C'^-g (uc0 r' + (1 - delta) P_I') / P_C'``, with
  ``beta = 1 / (1 + discount)``.

In a one-good closed economy with Cobb-Douglas production this is the
stochastic growth model, and the tests solve both and compare. The base year is
a steady state only when the SAM's investment equals depreciation of the stock
the user cost implies; :func:`dsgelab.steady_state` finds the steady state of
the model, which is another point, and a published table has to say which of
the two it reports.

dsgelab is an optional dependency: ``pip install "cgelab[dsge]"``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .closure import Closure
from .scenario import Scenario
from .solve import solve_period

__all__ = ["RamseyParams", "ramsey", "period_at", "announced"]


@dataclass
class RamseyParams:
    """Parameters dsgelab passes to the residuals. One per date on a path."""

    beta: float
    risk_aversion: float
    depreciation: float
    user_cost: float
    #: The scenario of this date and of the next: the Euler equation at ``t``
    #: solves the period at ``t + 1`` too, under the policy announced for it.
    scenario: Scenario = field(default_factory=Scenario)
    next_scenario: Scenario = field(default_factory=Scenario)
    rho_a: float = 0.0


def _require_dsgelab():
    try:
        import dsgelab
    except ImportError as exc:                      # pragma: no cover
        raise ImportError('cgelab.bridge needs dsgelab: pip install "cgelab[dsge]"') from exc
    return dsgelab


def period_at(econ, closure, scenario: Scenario, k: float, inv: float, a: float = 1.0, *,
              capital_factor: str = "f_cap", user_cost: float, warm=None):
    """The period equilibrium at capital stock ``k``, investment ``inv`` and TFP ``a``."""
    j = econ.factor(capital_factor)
    supply = (econ.fac_supply if scenario.factor_supply is None
              else np.asarray(scenario.factor_supply, float)).copy()
    supply[j] = user_cost * k
    tfp = np.ones(econ.n_act) if scenario.productivity is None else np.asarray(
        scenario.productivity, float)
    sc = dataclasses.replace(scenario, factor_supply=supply, productivity=tfp * a,
                             investment_level=float(inv))
    return solve_period(econ, closure, sc, warm_start=warm)


def _prices(res, capital):
    consumption = float(res.real["hh_real"])
    p_c = float(res.hh_commodity_spend.sum()) / consumption
    p_i = float(res.savings) / float(res.real["inv_real"])
    return consumption, p_c, p_i, float(res.factor_prices[capital])


def ramsey(econ, closure: Closure | None = None, *, capital_factor: str = "f_cap",
           depreciation: float = 0.05, discount: float = 0.05, risk_aversion: float = 1.0,
           scenario: Scenario | None = None, tfp_persistence: float | None = None):
    """Return ``(model, params, guess)`` for dsgelab.

    ``tfp_persistence`` adds a state ``a`` (Hicks-neutral TFP, AR(1) in logs)
    and a shock ``eps_a``; without it the model is deterministic in its states,
    which is what perfect-foresight experiments need. The closure's investment
    rule is replaced by ``"fixed_real"``: here the Euler equation sets it.
    """
    dsg = _require_dsgelab()
    closure = dataclasses.replace(closure or Closure(), investment="fixed_real")
    scenario = scenario or Scenario()
    capital = econ.factor(capital_factor)
    user_cost = discount + depreciation
    params = RamseyParams(beta=1.0 / (1.0 + discount), risk_aversion=risk_aversion,
                          depreciation=depreciation, user_cost=user_cost,
                          scenario=scenario, next_scenario=scenario,
                          rho_a=0.0 if tfp_persistence is None else float(tfp_persistence))
    shocked = tfp_persistence is not None
    cache: dict = {}
    warm: list = [None]

    def period(sc, k, inv, a):
        key = (id(sc), float(k), float(inv), float(a))
        hit = cache.get(key)
        if hit is None:
            if len(cache) > 20000:
                cache.clear()
            res = period_at(econ, closure, sc, k, inv, a, capital_factor=capital_factor,
                            user_cost=user_cost, warm=warm[0])
            warm[0] = res
            hit = _prices(res, capital)
            cache[key] = hit
        return hit

    def equations(nxt, cur, eps, p):
        a0 = cur.a if shocked else 1.0
        a1 = nxt.a if shocked else 1.0
        c0, pc0, pi0, _ = period(p.scenario, cur.k, cur.inv, a0)
        c1, pc1, pi1, r1 = period(p.next_scenario, nxt.k, nxt.inv, a1)
        g = p.risk_aversion
        out = [
            nxt.k - ((1.0 - p.depreciation) * cur.k + cur.inv),
            c0 ** (-g) * pi0 / pc0
            - p.beta * c1 ** (-g) * (p.user_cost * r1 + (1.0 - p.depreciation) * pi1) / pc1,
        ]
        if shocked:
            out.append(np.log(nxt.a) - p.rho_a * np.log(cur.a) - eps.eps_a)
        return out

    states = ["k", "a"] if shocked else ["k"]
    model = dsg.Model(states=states, jumps=["inv"], shocks=["eps_a"] if shocked else [],
                      equations=equations, name="ramsey_cge")
    k0 = float(econ.fac_supply[capital]) / user_cost
    guess = {"k": k0, "inv": depreciation * k0}
    if shocked:
        guess["a"] = 1.0
    return model, params, guess


def announced(params: RamseyParams, scenarios: Sequence[Scenario]) -> list:
    """Parameters for each date of a perfect-foresight path, one scenario per date.

    The last scenario holds from there on. Each date carries the next date's
    scenario as well, so the Euler equation at the switch sees the new policy.
    """
    seq = list(scenarios)
    out = []
    for t, sc in enumerate(seq):
        nxt = seq[t + 1] if t + 1 < len(seq) else sc
        out.append(dataclasses.replace(params, scenario=sc, next_scenario=nxt))
    return out
