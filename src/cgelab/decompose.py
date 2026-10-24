"""Subtotals: how much of a result each group of shocks accounts for.

A policy rarely moves one thing. A tariff reform comes with a world-price
change and a tax replacement, and the question that follows the result is how
much of it came from which. Removing one shock at a time does not answer it:
the parts then depend on the order of removal and do not add up to the total
once the model is nonlinear.

Harrison, Horridge and Pearson (2000) give the answer GEMPACK reports as
subtotals. Move every shock together along the straight line from the
reference to the alternative, ``x(t) = x0 + t (x1 - x0)``. At each point the
change in a result is the sum over shocks of its partial derivative times that
shock's step. Integrating each term separately over ``t`` splits the total
change into contributions that

* add up to the total exactly, because together they are the integral of
  ``dy/dt``;
* do not depend on any ordering of the shocks, because there is none;
* give each shock its own effect whenever the result is additive in them.

Here the integral is Gauss-Legendre over ``t`` and the partial derivatives are
central differences along each group's direction, each a solve warm-started
from the point on the path. ``residual`` is the total change computed directly
minus the sum of the contributions, the numerical error of both steps, and it is
reported rather than spread over the groups.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .scenario import Scenario
from .sensitivity import _as_series

__all__ = ["path_decomposition", "subtotals", "scenario_vector"]

# Numeric Scenario fields, and the value an absent one stands for.
_DEFAULTS = {
    "factor_supply": lambda e: e.fac_supply,
    "productivity": lambda e: np.ones(e.n_act),
    "product_tax": lambda e: e.ptax,
    "product_tax_firm": None,                # absent means "equal to product_tax"
    "consumer_wedge": lambda e: np.zeros((e.n_hh, e.n_com)),
    "government_wedge": lambda e: np.zeros((e.n_hh, e.n_com)),
    "transfers": lambda e: np.zeros(e.n_hh),
    "government_absorbs": lambda e: 0.0,
    "world_price": lambda e: np.ones(e.n_com),
    "tariff": lambda e: e.tm0,
    "output_tax": lambda e: e.tauz0,
    "export_price": lambda e: np.ones(e.n_com),
    "factor_ownership": lambda e: e.hh_facshare,
    "foreign_factor_share": lambda e: e.zeros_if_none("fac_row_share", e.n_fac),
    "carbon_price": lambda e: 0.0,
    "investment_level": lambda e: float(e.inv_dem.sum()),
}
_SCALARS = ("government_absorbs", "carbon_price", "investment_level")
_MAPPINGS = ("activity_scale", "fdi")


def path_decomposition(f: Callable[[dict], object], x0: Mapping[str, np.ndarray],
                       x1: Mapping[str, np.ndarray], groups: Mapping[str, Sequence[str]], *,
                       nodes: int = 6, step: float = 1e-4, solve_at=None) -> pd.DataFrame:
    """Decompose ``f(x1) - f(x0)`` over groups of the keys of ``x``.

    ``f`` maps a dict of arrays to a number, a mapping or a Series. Every key
    that differs between ``x0`` and ``x1`` must belong to exactly one group.
    ``solve_at(x)``, when given, is called once per node and its return value is
    passed to ``f`` as ``f(x, at)`` for warm starts.
    """
    keys = list(x0)
    if set(keys) != set(x1):
        raise ValueError("x0 and x1 must have the same keys")
    owner = {}
    for g, members in groups.items():
        for k in members:
            if k not in x0:
                raise KeyError(f"group {g!r} names {k!r}, which is not a shocked field")
            if k in owner:
                raise ValueError(f"{k!r} is in groups {owner[k]!r} and {g!r}")
            owner[k] = g
    x0 = {k: np.asarray(x0[k], dtype=float) for k in keys}
    x1 = {k: np.asarray(x1[k], dtype=float) for k in keys}
    moved = [k for k in keys if not np.array_equal(x0[k], x1[k])]
    orphan = [k for k in moved if k not in owner]
    if orphan:
        raise ValueError(f"{orphan} change between the two scenarios but are in no group; "
                         f"their effect would be left unattributed")
    delta = {k: x1[k] - x0[k] for k in keys}

    def at(t, group=None, h=0.0):
        out = {}
        for k in keys:
            v = x0[k] + t * delta[k]
            if group is not None and owner.get(k) == group:
                v = v + h * delta[k]
            out[k] = v
        return out

    def call(x, warm):
        return _as_series(f(x) if solve_at is None else f(x, warm))

    tau, weights = np.polynomial.legendre.leggauss(nodes)
    tau, weights = 0.5 * (tau + 1.0), 0.5 * weights        # onto [0, 1]
    contrib = {g: None for g in groups}
    for t, w in zip(tau, weights):
        warm = None if solve_at is None else solve_at(at(t))
        for g in groups:
            if not any(owner.get(k) == g and k in moved for k in keys):
                d = None
            else:
                up = call(at(t, g, step), warm)
                down = call(at(t, g, -step), warm)
                d = w * (up - down) / (2.0 * step)
            if d is not None:
                contrib[g] = d if contrib[g] is None else contrib[g] + d
    y0 = call(at(0.0), None if solve_at is None else solve_at(at(0.0)))
    y1 = call(at(1.0), None if solve_at is None else solve_at(at(1.0)))
    out = pd.DataFrame({g: (c if c is not None else 0.0 * y0) for g, c in contrib.items()})
    out["total"] = out[list(groups)].sum(axis=1)
    out["change"] = y1 - y0
    out["residual"] = out["change"] - out["total"]
    out.insert(0, "reference", y0)
    return out


def scenario_vector(econ, scenario: Scenario, *, firm_rate: bool | None = None) -> dict:
    """The numeric fields of a scenario as arrays, absent ones at what they stand for.

    Mapping fields (``activity_scale``, ``fdi``) become one entry per key,
    ``"activity_scale[a_own]"``. Used to put two scenarios on one straight line.
    ``product_tax_firm`` is left out unless the scenario sets it (or
    ``firm_rate`` asks for it): absent, it follows ``product_tax``.
    """
    sc = scenario
    if firm_rate is None:
        firm_rate = sc.product_tax_firm is not None
    out = {}
    for name, default in _DEFAULTS.items():
        value = getattr(sc, name)
        if name == "product_tax_firm":
            if not firm_rate:
                continue
            value = sc.product_tax if value is None else value
            value = econ.ptax if value is None else value
        elif value is None:
            value = default(econ)
        if value is None:
            continue                       # an instrument the base year did not calibrate
        out[name] = np.asarray(value, dtype=float)
    for name in _MAPPINGS:
        for key, value in getattr(sc, name).items():
            out[f"{name}[{key}]"] = np.asarray(float(value))
    return out


def _to_scenario(base: Scenario, x: dict, mapping_keys: dict, active: set) -> Scenario:
    # Only fields one of the two scenarios set: an absent field stays absent, so
    # an instrument the model refuses (export_price under fixed shares) is never
    # handed to it at its neutral value.
    changes = {k: v for k, v in x.items() if "[" not in k and k in active}
    for name in _SCALARS:
        if name in changes:
            changes[name] = float(changes[name])
    for name in _MAPPINGS:
        changes[name] = {key: float(x[f"{name}[{key}]"]) for key in mapping_keys[name]}
    return dataclasses.replace(base, **changes)


def _refuse_rules_that_differ(reference: Scenario, alternative: Scenario) -> None:
    """Every Scenario field is either on the path or must agree between the ends.

    A field this module does not know as numeric is treated as a rule; if it
    differs, the decomposition would solve both ends without it and report
    nothing, which is worse than an error.
    """
    fixed = [f.name for f in dataclasses.fields(Scenario)
             if f.name not in _DEFAULTS and f.name not in _MAPPINGS and f.name != "name"]
    differ = [n for n in fixed if getattr(reference, n) != getattr(alternative, n)]
    if differ:
        raise ValueError(f"{differ} differ between the scenarios and are not numbers; "
                         f"there is no path between them")


def subtotals(econ, reference: Scenario, alternative: Scenario,
              groups: Mapping[str, Sequence[str]], *, measure: Callable, closure=None,
              nodes: int = 6, step: float = 1e-4) -> pd.DataFrame:
    """Contributions of groups of scenario fields to the change in ``measure``.

    ``groups`` maps a label to Scenario field names (``"tariff"``,
    ``"world_price"``, ``"activity_scale[a_own]"``). Every field that differs
    between the two scenarios must be in a group. ``measure(period)`` returns a
    number, a mapping or a Series. Fields that are not numbers -- the recycling
    rule, the name -- must agree between the two scenarios: there is no straight
    line between two rules.

    Returns one row per output: ``reference`` (its value under the reference),
    one column per group, ``total`` (their sum), ``change`` (alternative minus
    reference, solved directly) and ``residual``.
    """
    from .solve import solve_period

    _refuse_rules_that_differ(reference, alternative)
    firm = reference.product_tax_firm is not None or alternative.product_tax_firm is not None
    x0 = scenario_vector(econ, reference, firm_rate=firm)
    x1 = scenario_vector(econ, alternative, firm_rate=firm)
    mapping_keys = {}
    for name in _MAPPINGS:
        keys = sorted(set(getattr(reference, name)) | set(getattr(alternative, name)))
        mapping_keys[name] = keys
        for key in keys:
            neutral = 1.0 if name == "activity_scale" else 0.0
            x0.setdefault(f"{name}[{key}]", np.asarray(neutral))
            x1.setdefault(f"{name}[{key}]", np.asarray(neutral))
    if set(x0) != set(x1):
        raise ValueError(f"the scenarios shock different instruments: "
                         f"{sorted(set(x0) ^ set(x1))}")
    # Fields named in a group are always shocked; a group naming a whole
    # mapping field takes all its keys.
    active = {n for n in _DEFAULTS
              if getattr(reference, n) is not None or getattr(alternative, n) is not None}
    for k in [k for k in x0 if "[" not in k and k not in active]:
        if not np.array_equal(x0[k], x1[k]):          # cannot happen; kept as a guard
            raise AssertionError(k)
        del x0[k], x1[k]
    # A group naming a whole mapping field takes all its keys. A field neither
    # scenario sets is not shocked, and its group contributes zero.
    known = set(_DEFAULTS) | set(_MAPPINGS)
    expanded = {}
    for g, members in groups.items():
        names = []
        for m in members:
            if m.partition("[")[0] not in known:
                raise KeyError(f"group {g!r} names {m!r}, which is not a numeric Scenario field")
            names += ([k for k in x0 if k.startswith(f"{m}[")] if m in _MAPPINGS
                      else [m] if m in x0 else [])
        expanded[g] = names

    def solve_at(x):
        return solve_period(econ, closure, _to_scenario(reference, x, mapping_keys, active))

    def f(x, warm):
        return measure(solve_period(econ, closure, _to_scenario(reference, x, mapping_keys, active),
                                    warm_start=warm))

    return path_decomposition(f, x0, x1, expanded, nodes=nodes, step=step, solve_at=solve_at)


# ---------------------------------------------------------------------------
# Welfare decomposition (Huff and Hertel, 2000)
# ---------------------------------------------------------------------------

def _rates(econ, period):
    """The instruments in force in a solved period, absent ones at their defaults."""
    sc = period.scenario
    ptax_a = econ.ptax if sc.product_tax is None else np.asarray(sc.product_tax, float)
    ptax_f = ptax_a if sc.product_tax_firm is None else np.asarray(sc.product_tax_firm, float)
    # Under a government target the adjusting tax's rates are the scenario's
    # times the factor the solver found; the decomposition must use those.
    closure = period.closure
    if closure.government_target is not None and closure.government_adjusts == "product_tax":
        ptax_a, ptax_f = ptax_a * period.tax_scale, ptax_f * period.tax_scale
    W = (np.zeros((econ.n_hh, econ.n_com)) if sc.consumer_wedge is None
         else np.asarray(sc.consumer_wedge, float))
    Wg = (np.zeros_like(W) if sc.government_wedge is None
          else np.asarray(sc.government_wedge, float))
    tm = econ.tm0 if sc.tariff is None else np.asarray(sc.tariff, float)
    tauz = econ.tauz0 if sc.output_tax is None else np.asarray(sc.output_tax, float)
    tfp = np.ones(econ.n_act) if sc.productivity is None else np.asarray(sc.productivity, float)
    scale = np.ones(econ.n_act)
    for name, mult in sc.activity_scale.items():
        scale[econ.activity(name) if isinstance(name, str) else int(name)] = float(mult)
    return dict(ptax_a=ptax_a, ptax_f=ptax_f, W=W, Wg=Wg, tm=tm, tauz=tauz, tfp=tfp,
                own=econ.hh_direct.sum(axis=0) * scale)


def _welfare_state(period) -> dict:
    """Quantities and prices the decomposition differentiates, from one solved period."""
    econ = period.econ
    r = _rates(econ, period)
    nu = period.closure.numeraire
    from .emissions import carbon_wedges
    carb_act, carb_hh = carbon_wedges(period)
    price = 0.0 if period.scenario.carbon_price is None else (
        float(period.scenario.carbon_price) * nu)
    em_act = carb_act * period.X / price if price else np.zeros(econ.n_act)
    carb_proc = price * econ.emissions.process if price else np.zeros(econ.n_act)
    fin = period.hh_quantity.sum(axis=0) + period.gov_quantity
    # An imperfectly mobile factor's endowment is its pools; its use is by
    # activity, at the activity's own price.
    mobile = np.ones(econ.n_fac, dtype=bool)
    mobile[list(econ.mobility)] = False
    pools = period.factor_pools or {}
    pool_w = np.concatenate([pools[econ.factors[i]]["price"] for i in econ.mobility] or [[]])
    pool_l = np.concatenate([pools[econ.factors[i]]["endowment"] for i in econ.mobility] or [[]])
    w_act = (period.factor_prices_by_activity if period.factor_prices_by_activity is not None
             else np.tile(period.factor_prices[:, None], (1, econ.n_act)))
    return dict(
        mobile=mobile, Wpool=pool_w, Fpool=pool_l, wa=w_act,
        Fa=period.factor_demand_by_activity,
        hh_q=period.hh_quantity, own=r["own"], gov_q=period.gov_quantity,
        inv_q=period.investment_quantity, fin=fin, firm=period.absorption - fin,
        X=period.X, Q=period.Q, M=period.imports, E=period.exports,
        # The export price in foreign currency is what exports actually fetch over the
        # exchange rate. Under CET that is the world price; under fixed export shares
        # exports sell at the producer price, so it moves with domestic costs, and
        # leaving that out of the terms of trade leaves the identity open.
        F=period.factor_demand, pWe=period.pe / period.exchange_rate,
        pWm=period.world_import_price,
        er=period.exchange_rate, FS=period.foreign_saving, row=period.factor_income_abroad,
        log_tfp=np.log(r["tfp"]),
        # prices and rates at this point
        eff=period.pq[None, :] * (1.0 + r["W"] + r["Wg"]) + carb_hh[None, :], pa=period.pa,
        pq=period.pq, carb_act=carb_act, carb_hh=carb_hh, carbon=price, em_act=em_act,
        carb_proc=carb_proc,
        pqf=period.pqf, pbase=period.pbase, pd=period.pd, w=period.factor_prices,
        ptax_a=r["ptax_a"], ptax_f=r["ptax_f"], wedge=r["W"] + r["Wg"], tm=r["tm"],
        tauz=r["tauz"], atax=econ.atax * nu,
    )


_WELFARE_TERMS = ["endowment", "technology", "allocative_product_tax",
                  "allocative_consumer_wedge", "allocative_activity_tax", "allocative_carbon_tax",
                  "allocative_tariff", "allocative_output_tax", "factor_reallocation",
                  "terms_of_trade", "external_account"]


def _welfare_integrand(s: dict, d: dict) -> dict:
    """Each term at one point: prices and rates from ``s``, rates of change from ``d``."""
    pm_b = s["er"] * s["pWm"]
    parts = {
        "households": float((s["eff"] * d["hh_q"]).sum()),
        "own_account": float(s["pa"] @ d["own"]),
        "government": float(s["pq"] @ d["gov_q"]),
        "investment": float(s["pqf"] @ d["inv_q"]),
    }
    out = {
        "absorption": sum(parts.values()),
        **{f"absorption_{k}": v for k, v in parts.items()},
        # A perfectly mobile factor: its price times the change in its use. An
        # imperfectly mobile one: each pool's price times the change in its
        # endowment.
        "endowment": float(s["w"][s["mobile"]] @ d["F"][s["mobile"]]
                           + s["Wpool"] @ d["Fpool"]),
        # What moving an imperfectly mobile factor between activities adds
        # beyond its endowment: the activity prices times the change in use,
        # less the endowment term. A CET frontier makes it zero (the envelope
        # theorem on the pool's revenue); a logit allocation does not, because
        # its wage gaps are not the cost of moving.
        "factor_reallocation": float((s["wa"][~s["mobile"]] * d["Fa"][~s["mobile"]]).sum()
                                     - s["Wpool"] @ d["Fpool"]),
        # Every input cost that productivity scales, the carbon tax on the fuel
        # among them; not the activity tax or process carbon, which are per unit
        # of output whatever the inputs.
        "technology": float(((s["pa"] - s["atax"] - s["carb_proc"]) * s["X"]) @ d["log_tfp"]),
        "allocative_product_tax": float((s["pbase"] * (s["ptax_f"] * d["firm"]
                                                       + s["ptax_a"] * d["fin"])).sum()),
        "allocative_consumer_wedge": float((s["wedge"] * s["pq"][None, :] * d["hh_q"]).sum()),
        "allocative_activity_tax": float(s["atax"] @ d["X"]),
        # The taxed quantity is the emission itself, so productivity that cuts
        # fuel per unit of output counts here as well as a change in output.
        "allocative_carbon_tax": float(s["carbon"] * d["em_act"].sum()
                                       + s["carb_hh"] @ d["hh_q"].sum(axis=0)),
        "allocative_tariff": (0.0 if s["tm"] is None
                              else float((s["tm"] * pm_b) @ d["M"])),
        "allocative_output_tax": (0.0 if s["tauz"] is None
                                  else float((s["tauz"] * s["pd"]) @ d["Q"])),
        "terms_of_trade": float(s["er"] * (s["E"] @ d["pWe"] - s["M"] @ d["pWm"])),
        "external_account": float(d["FS"] + (s["pWe"] @ s["E"] - s["pWm"] @ s["M"]) * d["er"]
                                  - d["row"]),
    }
    return out


def welfare_decomposition(econ, reference: Scenario, alternative: Scenario, *, closure=None,
                          nodes: int = 8, step: float = 1e-4) -> pd.Series:
    """Decompose the change in real absorption between two scenarios, Huff-Hertel style.

    The welfare measure is the Divisia volume of domestic absorption: household
    consumption (at what households pay), own-account purchases, government
    consumption and investment, each priced at the current prices along the
    path, ``integral of sum p dq``. Differentiating the GDP identity and the
    balance of payments splits it into

    * ``endowment`` -- factor prices times the change in factor use; for an
      imperfectly mobile factor, each pool's price times the change in its
      endowment;
    * ``factor_reallocation`` -- for an imperfectly mobile factor, its price in
      each activity times the change in its use there, less the endowment
      term. Zero under a CET frontier, which already prices the cost of moving;
      under a logit allocation, the gain from moving workers toward the
      activities that pay more;
    * ``technology`` -- gross cost times the change in log productivity;
    * ``allocative_*`` -- one term per instrument, its rate times the price of
      the taxed flow times the change in that flow: a tax that shrinks a taxed
      flow costs welfare, one that grows it gains (Harberger's triangles);
    * ``terms_of_trade`` -- the exchange rate times exports times the change in
      their price in foreign currency, less the same for imports, so the
      exchange rate itself does not count as a terms-of-trade change. Under
      fixed export shares exports sell at the producer price, and their foreign
      price moves with domestic costs;
    * ``external_account`` -- the change in foreign saving and its revaluation
      by the exchange rate, less factor income paid abroad.

    The terms are integrated along the straight path from ``reference`` to
    ``alternative``, as in :func:`subtotals`. ``residual`` is the absorption
    change minus the sum of the terms: zero up to the integration error when the
    identity holds, and the test of it. The measure is a Divisia volume, not
    the households' EV: it counts government and investment volumes, and it
    prices along the path rather than at the reference. The
    ``absorption_households``, ``_own_account``, ``_government`` and
    ``_investment`` rows split the measure by component; a tariff abolition
    can raise the households' part while public consumption, which loses the
    revenue, pulls the total down.
    """
    from .solve import solve_period

    _refuse_rules_that_differ(reference, alternative)
    firm = reference.product_tax_firm is not None or alternative.product_tax_firm is not None
    x0 = scenario_vector(econ, reference, firm_rate=firm)
    x1 = scenario_vector(econ, alternative, firm_rate=firm)
    mapping_keys = {}
    for name in _MAPPINGS:
        keys = sorted(set(getattr(reference, name)) | set(getattr(alternative, name)))
        mapping_keys[name] = keys
        for key in keys:
            neutral = 1.0 if name == "activity_scale" else 0.0
            x0.setdefault(f"{name}[{key}]", np.asarray(neutral))
            x1.setdefault(f"{name}[{key}]", np.asarray(neutral))
    active = {n for n in _DEFAULTS
              if getattr(reference, n) is not None or getattr(alternative, n) is not None}
    if firm:
        active.add("product_tax_firm")

    def solve(t, warm=None):
        x = {k: x0[k] + t * (x1[k] - x0[k]) for k in x0}
        return solve_period(econ, closure, _to_scenario(reference, x, mapping_keys, active),
                            warm_start=warm)

    tau, weights = np.polynomial.legendre.leggauss(nodes)
    tau, weights = 0.5 * (tau + 1.0), 0.5 * weights
    total = dict.fromkeys(["absorption", "absorption_households", "absorption_own_account",
                           "absorption_government", "absorption_investment"]
                          + _WELFARE_TERMS, 0.0)
    for t, w in zip(tau, weights):
        mid = solve(t)
        s = _welfare_state(mid)
        up, down = _welfare_state(solve(t + step, mid)), _welfare_state(solve(t - step, mid))
        d = {k: (np.asarray(up[k], float) - np.asarray(down[k], float)) / (2 * step)
             for k in ("hh_q", "own", "gov_q", "inv_q", "fin", "firm", "X", "Q", "M", "E",
                       "F", "Fa", "Fpool", "pWe", "pWm", "er", "FS", "row", "log_tfp",
                       "em_act")}
        for k, v in _welfare_integrand(s, d).items():
            total[k] += w * v
    out = pd.Series(total, name="welfare")
    out["sum_of_terms"] = float(out[_WELFARE_TERMS].sum())
    out["residual"] = out["absorption"] - out["sum_of_terms"]
    return out
