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
}
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
    if "government_absorbs" in changes:
        changes["government_absorbs"] = float(changes["government_absorbs"])
    for name in _MAPPINGS:
        changes[name] = {key: float(x[f"{name}[{key}]"]) for key in mapping_keys[name]}
    return dataclasses.replace(base, **changes)


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

    fixed = [f.name for f in dataclasses.fields(Scenario)
             if f.name not in _DEFAULTS and f.name not in _MAPPINGS and f.name != "name"]
    differ = [n for n in fixed if getattr(reference, n) != getattr(alternative, n)]
    if differ:
        raise ValueError(f"{differ} differ between the scenarios and are not numbers; "
                         f"there is no path between them")
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
