"""A generic closure swap, in the GEMPACK manner, for one period.

A standard closure makes the scenario's instruments exogenous and reads the model's variables
off the solution. A swap reverses some of those roles: a variable is set to an observed or
desired value, and an instrument that would otherwise be given becomes the unknown that
achieves it. Exports of a commodity against its world export price, employment against labour
supply, real GDP against a common productivity, a government balance against a tax rate.

The period equations stay intact. ``solve_swapped`` wraps :func:`cgelab.solve_period` in a root
over the swapped instruments, so every closure, scenario field and check of the period solver
holds at the answer. ``calibrate_historical`` is the recursive case with technology and
preference shifters; this is the general one, in a single period.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import root

from .calibrate import Economy
from .closure import Closure
from .scenario import Scenario
from .solve import PeriodResult, solve_period

__all__ = ["Swapped", "solve_swapped"]

# Instruments: the Scenario field, the accounts its entries follow, and whether it is a
# positive multiplier (solved in logs) or a level (a rate, a transfer, an amount).
_INSTRUMENTS = {
    "productivity": ("acts", "log"), "factor_supply": ("factors", "log"),
    "world_price": ("coms", "log"), "export_price": ("coms", "log"),
    "product_tax": ("coms", "level"), "product_tax_firm": ("coms", "level"),
    "tariff": ("coms", "level"), "output_tax": ("coms", "level"),
    "export_tax": ("coms", "level"), "transfers": ("hhs", "level"),
    "investment_level": (None, "log"), "carbon_price": (None, "level"),
    "government_absorbs": (None, "level"),
}
# Variables: the PeriodResult attribute and the accounts it follows.
_VARIABLES = {
    "X": "acts", "pa": "acts", "Q": "coms", "pd": "coms", "pq": "coms", "imports": "coms",
    "exports": "coms", "pe": "coms", "p_domestic": "coms", "absorption": "coms",
    "factor_prices": "factors", "factor_demand": "factors", "hh_income": "hhs",
    "exchange_rate": None, "employment": None, "unemployment": None, "real_wage": None,
    "savings": None, "foreign_saving": None, "household_saving": None,
}
_NAME = re.compile(r"^(?P<field>[A-Za-z_][A-Za-z0-9_.]*)(?:\[(?P<label>[^\]]+)\])?$")


@dataclass
class Swapped:
    """A period solved under a swapped closure.

    ``period`` is the solution; ``instruments`` the value each swapped instrument took (a
    common multiplier for ``field[*]``); ``scenario`` the scenario that, solved under the
    standard closure, gives the same period; ``achieved`` each swapped variable at the solution
    against its ``target``.
    """

    period: PeriodResult
    instruments: pd.Series
    scenario: Scenario
    achieved: pd.DataFrame
    iterations: int = 0
    econ: Economy = field(default=None, repr=False)


def _accounts(econ, axis):
    return {"acts": econ.acts, "coms": econ.coms, "factors": econ.factors,
            "hhs": econ.hhs}[axis]


def _default(econ, name):
    from .decompose import _DEFAULTS
    if name == "export_tax":
        value = econ.te0
    elif name == "product_tax_firm":
        value = econ.ptax
    elif name == "transfers":
        value = np.zeros(econ.n_hh)
    else:
        value = _DEFAULTS[name](econ)
    if value is None:
        raise ValueError(f"{name!r} cannot be swapped in: the instrument was not calibrated in "
                         f"this economy")
    return np.array(value, dtype=float) if np.ndim(value) else float(value)


def _parse_instrument(econ, name):
    m = _NAME.match(name)
    if not m or m["field"] not in _INSTRUMENTS:
        raise KeyError(f"instrument {name!r} is not one of "
                       f"{sorted(_INSTRUMENTS)}, written 'field[account]' or 'field[*]'")
    fname, label = m["field"], m["label"]
    axis, scale = _INSTRUMENTS[fname]
    if axis is None:
        if label is not None:
            raise KeyError(f"{fname!r} is a scalar; write it without an account")
        return fname, None, scale
    if label is None:
        raise KeyError(f"{fname!r} has one entry per account: write {fname}[account] or "
                       f"{fname}[*] for a common multiplier")
    if label == "*":
        return fname, "*", scale
    names = _accounts(econ, axis)
    if label not in names:
        raise KeyError(f"{label!r} is not one of {fname}'s accounts: {list(names)}")
    return fname, names.index(label), scale


def _measure(econ, target):
    """A callable on a PeriodResult for a target name or callable."""
    if callable(target):
        return target
    if target == "real_gdp":
        from .results import real_gdp
        return lambda p: float(real_gdp(p)["expenditure"])
    if target.startswith("government."):
        key = target.split(".", 1)[1]
        return lambda p: float(p.government[key])
    m = _NAME.match(target)
    if not m or m["field"] not in _VARIABLES:
        raise KeyError(f"variable {target!r} is not one of {sorted(_VARIABLES)}, "
                       f"'real_gdp', 'government.<key>' or a callable")
    attr, label = m["field"], m["label"]
    axis = _VARIABLES[attr]
    if axis is None:
        if label is not None:
            raise KeyError(f"{attr!r} is a scalar; write it without an account")
        return lambda p: float(getattr(p, attr))
    names = _accounts(econ, axis)
    if label not in names:
        raise KeyError(f"{target!r}: write {attr}[account] with an account of {list(names)}")
    k = names.index(label)
    return lambda p: float(np.asarray(getattr(p, attr))[k])


def solve_swapped(econ: Economy, closure: Closure | None = None,
                  scenario: Scenario | None = None, *,
                  exogenous: Mapping[str | Callable, float], endogenous: Sequence[str],
                  start: Mapping[str, float] | None = None, tol: float = 1e-11,
                  backend: str = "scipy") -> Swapped:
    """Solve a period with variables made exogenous and instruments made endogenous.

    ``exogenous`` maps each variable to the value it must take: a name such as
    ``"exports[c_ind]"``, ``"X[a_agr]"``, ``"employment"``, ``"real_gdp"`` (expenditure side),
    ``"government.saving"``, or any callable on a :class:`PeriodResult` (its key in the
    output is then its position). ``endogenous`` lists as many instruments, Scenario entries
    such as ``"productivity[a_agr]"``, ``"world_price[c_ind]"``, ``"tariff[c_agr]"``,
    ``"factor_supply[f_lab]"``, ``"investment_level"``, or ``"field[*]"`` for one common
    multiplier on every entry (on the scenario's values, or the base year's). Positive
    multipliers are solved in logs, rates and amounts in levels. An instrument that does not
    move the targets under the closure is refused. ``start`` gives first guesses by instrument; absent, the scenario's values.
    """
    closure = closure or Closure()
    base = scenario or Scenario()
    if len(exogenous) != len(endogenous):
        raise ValueError(f"a swap exchanges one for one: {len(exogenous)} variable(s) made "
                         f"exogenous against {len(endogenous)} instrument(s)")
    if not endogenous:
        raise ValueError("solve_swapped needs at least one swap; use solve_period otherwise")
    if len(set(endogenous)) != len(endogenous):
        raise ValueError(f"an instrument is swapped in twice: {list(endogenous)}")
    parsed = [_parse_instrument(econ, n) for n in endogenous]
    for (fname, k, _), name in zip(parsed, endogenous):
        if k == "*" and any(f == fname and kk not in ("*",) for f, kk, _ in parsed):
            raise ValueError(f"{name!r} and an entry of the same field cannot both be swapped in")
    labels = [t if isinstance(t, str) else f"variable {i}" for i, t in enumerate(exogenous)]
    measures = [_measure(econ, t) for t in exogenous]
    targets = np.array([float(v) for v in exogenous.values()])
    if not np.isfinite(targets).all():
        raise ValueError("every exogenous value must be a finite number")
    current = {}
    for fname, _, _ in parsed:
        own = getattr(base, fname)
        current[fname] = (_default(econ, fname) if own is None
                          else (np.array(own, float) if np.ndim(own) else float(own)))

    def scenario_at(x):
        values = {f: (v.copy() if isinstance(v, np.ndarray) else v) for f, v in current.items()}
        for value, (fname, k, scale) in zip(x, parsed):
            v = float(np.exp(value)) if scale == "log" else float(value)
            if k is None:
                values[fname] = v
            elif k == "*":
                values[fname] = values[fname] * v
            else:
                values[fname][k] = v
        return base.with_(**values)

    x0 = []
    for name, (fname, k, scale) in zip(endogenous, parsed):
        given = (start or {}).get(name)
        if k == "*":
            v = 1.0
        elif k is None:
            v = current[fname]
        else:
            v = current[fname][k]
        v = float(given) if given is not None else float(v)
        if scale == "log" and not v > 0:
            raise ValueError(f"{name!r} is a positive multiplier; its start must be positive")
        x0.append(np.log(v) if scale == "log" else v)
    scale_t = np.maximum(np.abs(targets), 1.0)
    memo: dict = {"warm": None, "n": 0}

    def residual(x):
        memo["n"] += 1
        period = solve_period(econ, closure, scenario_at(x), warm_start=memo["warm"],
                              backend=backend)
        memo["warm"], memo["period"] = period, period
        return (np.array([m(period) for m in measures]) - targets) / scale_t

    try:
        sol = root(residual, np.array(x0), method="hybr", tol=tol)
        final = residual(sol.x)
    except (RuntimeError, ValueError) as err:
        raise RuntimeError(
            f"solve_swapped did not reach the targets: a period failed on the way, after "
            f"{memo['n']} period solves. A target the instruments cannot reach takes the "
            f"model where it has no equilibrium.") from err
    # An instrument the closure does not read (investment_level under savings-driven
    # investment, a world export price under fixed export shares) leaves the targets where they
    # are: the root is then any value, or none, and either would be reported as solved.
    for j, name in enumerate(endogenous):
        bumped = np.array(sol.x, dtype=float)
        bumped[j] += 1e-4 * max(1.0, abs(bumped[j]))
        if np.max(np.abs(residual(bumped) - final)) < 1e-10:
            raise ValueError(
                f"{name!r} does not move {labels} under this closure and scenario: the swap is "
                f"not identified")
    final = residual(sol.x)
    if not np.max(np.abs(final)) <= max(tol * 100, 1e-9):
        raise RuntimeError(
            f"solve_swapped did not reach the targets: worst relative gap "
            f"{np.max(np.abs(final)):.3g} after {memo['n']} period solves ({sol.message}). "
            f"A target the instruments cannot reach, or two that pull the same way, has no "
            f"solution.")
    period = memo["period"]
    values = [float(np.exp(v)) if s == "log" else float(v) for v, (_, _, s) in zip(sol.x, parsed)]
    achieved = pd.DataFrame({"target": targets,
                             "achieved": [m(period) for m in measures]}, index=labels)
    return Swapped(period=period, instruments=pd.Series(values, index=list(endogenous),
                                                        name="instrument"),
                   scenario=scenario_at(sol.x), achieved=achieved, iterations=memo["n"],
                   econ=econ)
