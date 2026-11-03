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

Several capital stocks (one per region, or per sector) are several states,
each with its own investment jump, its own depreciation and its own Euler
equation. They are bought with the same investment bundle, so ``P_I`` is
common and arbitrage runs through the rental prices: in a steady state every
``r_i`` equals ``P_I``. Total real investment is the sum of the jumps.

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
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

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
    #: One value, or one per capital stock in the order of ``capital_factor``.
    depreciation: float | tuple
    user_cost: float | tuple
    #: The scenario of this date and of the next: the Euler equation at ``t``
    #: solves the period at ``t + 1`` too, under the policy announced for it.
    scenario: Scenario = field(default_factory=Scenario)
    next_scenario: Scenario = field(default_factory=Scenario)
    rho_a: float = 0.0
    #: The date on the path, set by :func:`announced`; it names the date when a
    #: period equilibrium fails. None when one set of parameters serves every date.
    date: int | None = None


def _require_dsgelab():
    try:
        import dsgelab
    except ImportError as exc:                      # pragma: no cover
        raise ImportError('cgelab.bridge needs dsgelab: pip install "cgelab[dsge]"') from exc
    return dsgelab


def period_at(econ, closure, scenario: Scenario, k, inv: float, a: float = 1.0, *,
              capital_factor: str | Sequence[str] = "f_cap", user_cost, warm=None):
    """The period equilibrium at capital stock ``k``, investment ``inv`` and TFP ``a``.

    With several capital factors, ``k`` and ``user_cost`` hold one value per
    factor and ``inv`` is total real investment.
    """
    factors = [capital_factor] if isinstance(capital_factor, str) else list(capital_factor)
    stocks = np.atleast_1d(np.asarray(k, float))
    costs = np.broadcast_to(np.asarray(user_cost, float), stocks.shape)
    if stocks.size != len(factors):
        raise ValueError(f"{stocks.size} capital stocks for {len(factors)} capital factors")
    supply = (econ.fac_supply if scenario.factor_supply is None
              else np.asarray(scenario.factor_supply, float)).copy()
    for name, stock, cost in zip(factors, stocks, costs):
        supply[econ.factor(name)] = cost * stock
    tfp = np.ones(econ.n_act) if scenario.productivity is None else np.asarray(
        scenario.productivity, float)
    sc = dataclasses.replace(scenario, factor_supply=supply, productivity=tfp * a,
                             investment_level=float(inv))
    return solve_period(econ, closure, sc, warm_start=warm)


def _prices(res, capitals):
    consumption = float(res.real["hh_real"])
    p_c = float(res.hh_commodity_spend.sum()) / consumption
    p_i = float(res.savings) / float(res.real["inv_real"])
    return consumption, p_c, p_i, np.asarray(res.factor_prices, float)[capitals]


def _changed(sc: Scenario) -> list:
    """The scenario's fields that differ from a default Scenario, by name."""
    default = Scenario()
    out = []
    for f in dataclasses.fields(sc):
        mine, base = getattr(sc, f.name), getattr(default, f.name)
        if mine is None and base is None:
            continue
        if isinstance(mine, np.ndarray) or isinstance(base, np.ndarray):
            same = (mine is not None and base is not None
                    and np.array_equal(np.asarray(mine), np.asarray(base)))
        else:
            try:
                same = bool(mine == base)
            except (TypeError, ValueError):
                same = False
        if not same:
            out.append(f.name)
    return out


def _where(p, which, k, inv, a, sc, names) -> str:
    """Where on the path a period failed, for the note on the solver's error."""
    date = "date unknown (one RamseyParams for every date)" if p.date is None else (
        f"date {p.date + (which == 'next')}")
    stocks = ", ".join(f"{n}={float(v):.6g}" for n, v in zip(names, np.atleast_1d(k)))
    changed = _changed(sc)
    return (f"cgelab.bridge: the period equilibrium failed at {date}, the "
            f"{which} period of the Euler equation, with {stocks}, total investment "
            f"{float(inv):.6g} and TFP {float(a):.6g}; scenario {sc.name!r}"
            + (f" sets {changed}" if changed else " is the base year") + ".")


def _tag(factor: str) -> str:
    return re.sub(r"\W", "_", factor)


def ramsey(econ, closure: Closure | None = None, *,
           capital_factor: str | Sequence[str] = "f_cap",
           depreciation: float | Mapping[str, float] = 0.05, discount: float = 0.05,
           risk_aversion: float = 1.0, scenario: Scenario | None = None,
           tfp_persistence: float | None = None):
    """Return ``(model, params, guess)`` for dsgelab.

    ``tfp_persistence`` adds a state ``a`` (Hicks-neutral TFP, AR(1) in logs)
    and a shock ``eps_a``; without it the model is deterministic in its states,
    which is what perfect-foresight experiments need. The closure's investment
    rule is replaced by ``"fixed_real"``: here the Euler equation sets it.

    One capital factor gives the state ``k`` and the jump ``inv``. A sequence
    of them, such as ``["r1:f_cap", "r2:f_cap"]``, gives a state ``k_<factor>``
    and a jump ``inv_<factor>`` for each, with every character that is not a
    letter, digit or underscore in the factor's name replaced by ``_``
    (``k_r1_f_cap``). ``depreciation`` is then one rate for all of them or a
    mapping from factor to rate.
    """
    dsg = _require_dsgelab()
    closure = dataclasses.replace(closure or Closure(), investment="fixed_real")
    scenario = scenario or Scenario()
    single = isinstance(capital_factor, str)
    factors = [capital_factor] if single else list(capital_factor)
    if not factors or len(set(factors)) != len(factors):
        raise ValueError(f"capital_factor must name distinct factors, got {factors}")
    capitals = np.array([econ.factor(f) for f in factors])
    if isinstance(depreciation, Mapping):
        missing = [f for f in factors if f not in depreciation]
        if missing:
            raise KeyError(f"depreciation has no rate for {missing}")
        delta = np.array([float(depreciation[f]) for f in factors])
    else:
        delta = np.full(len(factors), float(depreciation))
    user_cost = discount + delta
    if single:
        k_names, inv_names = ["k"], ["inv"]
    else:
        k_names = [f"k_{_tag(f)}" for f in factors]
        inv_names = [f"inv_{_tag(f)}" for f in factors]
        if len(set(k_names)) != len(k_names):
            raise ValueError(f"capital factors {factors} give clashing state names {k_names}")
    as_param = (lambda v: float(v[0])) if single else (lambda v: tuple(float(x) for x in v))
    params = RamseyParams(beta=1.0 / (1.0 + discount), risk_aversion=risk_aversion,
                          depreciation=as_param(delta), user_cost=as_param(user_cost),
                          scenario=scenario, next_scenario=scenario,
                          rho_a=0.0 if tfp_persistence is None else float(tfp_persistence))
    shocked = tfp_persistence is not None
    cache: dict = {}
    warm: list = [None]

    def period(p, which, k, inv, a):
        sc = p.scenario if which == "current" else p.next_scenario
        key = (id(sc), tuple(float(x) for x in k), float(inv), float(a))
        hit = cache.get(key)
        if hit is None:
            if len(cache) > 20000:
                cache.clear()
            try:
                res = period_at(econ, closure, sc, k, inv, a, capital_factor=factors,
                                user_cost=np.asarray(p.user_cost, float), warm=warm[0])
            except (ValueError, RuntimeError) as exc:
                exc.add_note(_where(p, which, k, inv, a, sc, k_names))
                raise
            warm[0] = res
            hit = _prices(res, capitals)
            cache[key] = hit
        return hit

    def equations(nxt, cur, eps, p):
        a0 = cur.a if shocked else 1.0
        a1 = nxt.a if shocked else 1.0
        k0 = np.array([getattr(cur, n) for n in k_names])
        k1 = np.array([getattr(nxt, n) for n in k_names])
        inv0 = np.array([getattr(cur, n) for n in inv_names])
        inv1 = np.array([getattr(nxt, n) for n in inv_names])
        d = np.broadcast_to(np.asarray(p.depreciation, float), k0.shape)
        uc = np.broadcast_to(np.asarray(p.user_cost, float), k0.shape)
        c0, pc0, pi0, _ = period(p, "current", k0, inv0.sum(), a0)
        c1, pc1, pi1, r1 = period(p, "next", k1, inv1.sum(), a1)
        g = p.risk_aversion
        out = list(k1 - ((1.0 - d) * k0 + inv0))
        out += list(c0 ** (-g) * pi0 / pc0
                    - p.beta * c1 ** (-g) * (uc * r1 + (1.0 - d) * pi1) / pc1)
        if shocked:
            out.append(np.log(nxt.a) - p.rho_a * np.log(cur.a) - eps.eps_a)
        return out

    states = k_names + (["a"] if shocked else [])
    model = dsg.Model(states=states, jumps=inv_names, shocks=["eps_a"] if shocked else [],
                      equations=equations, name="ramsey_cge")
    k_base = np.asarray(econ.fac_supply, float)[capitals] / user_cost
    guess = {n: float(v) for n, v in zip(k_names, k_base)}
    guess.update({n: float(d * v) for n, d, v in zip(inv_names, delta, k_base)})
    if shocked:
        guess["a"] = 1.0
    return model, params, guess


def announced(params: RamseyParams, scenarios: Sequence[Scenario]) -> list:
    """Parameters for each date of a perfect-foresight path, one scenario per date.

    The last scenario holds from there on. Each date carries the next date's
    scenario as well, so the Euler equation at the switch sees the new policy,
    and its own date, so a period that fails is located on the path.
    """
    seq = list(scenarios)
    out = []
    for t, sc in enumerate(seq):
        nxt = seq[t + 1] if t + 1 < len(seq) else sc
        out.append(dataclasses.replace(params, scenario=sc, next_scenario=nxt, date=t))
    return out
