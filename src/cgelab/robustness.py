"""Robustness: the same experiment under alternative closures, and results in money.

A published CGE result is read against three things a single solve does not
show: how it moves with the elasticities (:mod:`cgelab.sensitivity`), how it
moves with the closure, and how large it is in money rather than in percent.
This module holds the last two.

:func:`closure_sweep` solves one scenario under a list of closures. A closure is
a model of who bears an adjustment, and two defensible closures can give
results of opposite sign; a result that holds under every closure in the list is
worth more than one that holds under one. A combination the package refuses,
or one that does not solve, is reported with its reason instead of stopping the
sweep, because which combinations are impossible is part of the answer.

:func:`in_money` turns a counterfactual into levels: final demand, factor
income, household income and the government account, at current prices and at
base-year prices, in a unit the user picks.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .closure import Closure
from .results import _base_components, _components
from .scenario import Scenario
from .sensitivity import _as_dict

__all__ = ["ClosureSweep", "closure_sweep", "closure_grid", "in_money"]


# ---------------------------------------------------------------------------
# Closures
# ---------------------------------------------------------------------------

def closure_label(closure: Closure | Mapping) -> str:
    """The fields that differ from the default ``Closure()``, as ``"field=value, ..."``."""
    default = Closure()
    if isinstance(closure, Mapping):
        changed = {k: v for k, v in closure.items() if getattr(default, k, object()) != v}
    else:
        changed = {f.name: getattr(closure, f.name) for f in dataclasses.fields(Closure)
                   if getattr(closure, f.name) != getattr(default, f.name)}
    if not changed:
        return "default"
    return ", ".join(f"{k}={v}" for k, v in changed.items())


def closure_grid(**options: Sequence) -> dict:
    """Every combination of the given ``Closure`` fields, labelled.

    ``closure_grid(investment=["savings_driven", "fixed_real"],
    foreign=["fixed_saving", "fixed_rate"])`` gives four closures. The values
    are the keyword arguments of each closure, not ``Closure`` objects, so that
    a combination ``Closure`` itself refuses reaches :func:`closure_sweep` and
    is reported there with its reason. A field given one value applies to every
    combination.
    """
    names = [f.name for f in dataclasses.fields(Closure)]
    unknown = [k for k in options if k not in names]
    if unknown:
        raise TypeError(f"{unknown} are not Closure fields")
    if not options:
        return {"default": {}}
    keys = list(options)
    lists = [list(v) if isinstance(v, (list, tuple)) else [v] for v in options.values()]
    grid = {}
    for combo in itertools.product(*lists):
        kwargs = dict(zip(keys, combo))
        grid[closure_label(kwargs)] = kwargs
    return grid


@dataclass
class ClosureSweep:
    """One scenario solved under several closures."""

    #: One row per closure, one column per output of ``measure``. A closure that
    #: failed has a row of NaN.
    table: pd.DataFrame
    #: The largest Walras residual of the two solves, by closure.
    walras: pd.Series
    #: Whether both solves converged, by closure.
    converged: pd.Series
    #: ``{label: reason}`` for each closure the package refused or could not solve.
    failed: dict = field(default_factory=dict)
    #: ``{label: Closure}`` for each closure that was built.
    closures: dict = field(default_factory=dict, repr=False)

    def range(self) -> pd.DataFrame:
        """Each output's minimum, maximum and spread over the closures that solved.

        ``argmin`` and ``argmax`` name the closures at the two ends, and
        ``same_sign`` says whether every closure agrees on the direction.
        """
        ok = self.table.dropna(how="all")
        if ok.empty:
            raise ValueError("no closure solved")
        out = pd.DataFrame({"min": ok.min(), "max": ok.max()})
        out["spread"] = out["max"] - out["min"]
        out["argmin"] = ok.idxmin()
        out["argmax"] = ok.idxmax()
        out["same_sign"] = (np.sign(ok).nunique() == 1) & (ok != 0).all()
        return out

    def __repr__(self) -> str:
        return (f"ClosureSweep({len(self.table)} closures, {len(self.failed)} failed, "
                f"{self.table.shape[1]} outputs)")


def closure_sweep(econ, closures, *, measure: Callable, scenario=None, reference=None,
                  years: Sequence | None = None, growth=None, economy=None) -> ClosureSweep:
    """Solve the reference and the scenario under each closure, and measure the change.

    Parameters
    ----------
    closures : mapping or sequence
        ``{label: closure}`` or a list of closures. A closure is a ``Closure``
        or the keyword arguments of one (what :func:`closure_grid` returns). A
        list is labelled by the fields that differ from the default closure.
    measure : callable
        ``measure(reference, alternative)`` -> a number, a mapping or a Series:
        the same signature as in :func:`cgelab.sensitivity.spec_sweep`. With
        ``years`` it receives two ``PathResult`` objects.
    scenario, reference : Scenario or callable
        The alternative and the reference. The reference defaults to no
        change, which is the base year in a period and the growth path in a
        path. With ``years`` either may be ``f(year, t) -> Scenario``.
    years, growth, economy
        Given ``years``, both are paths of :func:`cgelab.run_path`.

    A closure the package refuses (an invalid combination of fields), or under
    which the model does not solve, gets a row of NaN, ``converged`` False and
    its reason in ``failed``; the sweep goes on.
    """
    from .dynamics import run_path
    from .solve import solve_period

    if isinstance(closures, Mapping):
        items = list(closures.items())
    else:
        items = [(closure_label(c), c) for c in closures]
        labels = [k for k, _ in items]
        dup = sorted({k for k in labels if labels.count(k) > 1})
        if dup:
            raise ValueError(f"closures {dup} appear twice in the list")
    if not items:
        raise ValueError("no closures to sweep")

    rows, walras, converged, failed, built = {}, {}, {}, {}, {}
    for label, spec in items:
        try:
            closure = spec if isinstance(spec, Closure) else Closure(**dict(spec))
            built[label] = closure
            if years is None:
                ref = solve_period(econ, closure, reference)
                alt = solve_period(econ, closure, scenario, warm_start=ref)
                periods = [ref, alt]
            else:
                kw = dict(years=years, growth=growth, economy=economy)
                ref = run_path(econ, closure, scenario=reference, **kw)
                alt = run_path(econ, closure, scenario=scenario, **kw)
                periods = ref.periods + alt.periods
            rows[label] = _as_dict(measure(ref, alt))
            walras[label] = max(abs(float(p.walras)) for p in periods)
            converged[label] = all(p.converged for p in periods)
        except (ValueError, RuntimeError, KeyError, TypeError, np.linalg.LinAlgError) as err:
            failed[label] = f"{type(err).__name__}: {err}"
            rows[label] = {}
            walras[label] = np.nan
            converged[label] = False
    order = [k for k, _ in items]
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(order)
    table.index.name = "closure"
    return ClosureSweep(table=table, walras=pd.Series(walras).reindex(order),
                        converged=pd.Series(converged).reindex(order).astype(bool),
                        failed=failed, closures=built)


# ---------------------------------------------------------------------------
# Results in money
# ---------------------------------------------------------------------------

_COLUMNS = ["reference", "alternative", "change", "reference_base_prices",
            "alternative_base_prices", "change_base_prices", "pct_base_prices"]


def _gdp_deflator(period) -> float:
    """Paasche GDP deflator against the calibrated base year: nominal over real GDP."""
    cur, base = _components(period), _base_components(period.econ)
    sign = {"imports": -1.0}
    nominal = sum(sign.get(k, 1.0) * float(p @ q) for k, (p, q) in cur.items())
    real = sum(sign.get(k, 1.0) * float(base[k][0] @ q) for k, (p, q) in cur.items())
    return nominal / real


def _base_price_index(econ) -> np.ndarray:
    """Each household's cost-of-living index at the calibrated base year.

    The solver's index is in purchaser prices, so at the base year it is the
    product tax the household pays, not one.
    """
    from .solve import _marginal_budget_shares, _price_index
    p0 = np.broadcast_to(1.0 + econ.ptax, (econ.n_hh, econ.n_com))
    return _price_index(econ, p0, _marginal_budget_shares(econ))


def _levels(period) -> dict:
    """``{(block, item): (current, base_prices)}`` for one solved period."""
    econ = period.econ
    cur, base = _components(period), _base_components(econ)
    out = {}
    gdp_cur = gdp_base = 0.0
    for k, (p, q) in cur.items():
        s = -1.0 if k == "imports" else 1.0
        c, b = float(p @ q), float(base[k][0] @ q)
        out[("final_demand", k)] = (c, b)
        gdp_cur += s * c
        gdp_base += s * b
    out[("final_demand", "gdp")] = (gdp_cur, gdp_base)
    # Factor income at the prices each factor earns, and its services at the
    # base-year price of one.
    for i, f in enumerate(econ.factors):
        out[("factor_income", f)] = (float(period.factor_prices[i] * period.factor_demand[i]),
                                     float(period.factor_demand[i]))
    # Households: income they can spend (net of direct tax and of payments made
    # straight to an activity), and the same deflated by their own price index
    # against the base year's. The change of the second is the EV valued at
    # base-year prices.
    tax = (np.zeros(econ.n_hh) if period.hh_direct_tax is None
           else np.asarray(period.hh_direct_tax, float))
    spend = np.asarray(period.hh_income, float) - np.asarray(period.hh_direct_outlay, float) - tax
    index = np.asarray(period.real["hh_price_index"], float) / _base_price_index(econ)
    for i, h in enumerate(econ.hhs):
        out[("households", h)] = (float(spend[i]), float(spend[i] / index[i]))
    # Government: nominal flows, in volume through the GDP deflator.
    deflator = _gdp_deflator(period)
    g = period.government
    for k in ("revenue", "consumption", "transfers", "saving"):
        out[("government", k)] = (float(g[k]), float(g[k]) / deflator)
    return out


def _money_table(ref, alt, unit) -> pd.DataFrame:
    a, b = _levels(ref), _levels(alt)
    rows = {}
    for key in a:
        (rc, rb), (ac, ab) = a[key], b[key]
        rows[key] = [rc, ac, ac - rc, rb, ab, ab - rb,
                     100.0 * (ab / rb - 1.0) if rb != 0 else np.nan]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=_COLUMNS)
    out.index = pd.MultiIndex.from_tuples(out.index, names=["block", "item"])
    out[_COLUMNS[:-1]] = out[_COLUMNS[:-1]] * float(unit)
    return out


def in_money(reference, alternative, *, unit: float = 1.0, label: str | None = None,
             years: Sequence | None = None) -> pd.DataFrame:
    """A counterfactual in levels: at current prices, at base-year prices, and the change.

    ``reference`` and ``alternative`` are two solved periods, or two paths
    (``PathResult``), in which case every year they share is reported, or the
    years in ``years``.

    Rows, by block: ``final_demand`` (consumption, government, investment,
    exports, imports and GDP by expenditure), ``factor_income`` (each factor's
    income; at base-year prices, its services), ``households`` (income net of
    direct tax and of payments straight to an activity; at base-year prices,
    over the household's own price index, so the change is the EV at base-year
    prices) and ``government`` (revenue, consumption, transfers and saving; at
    base-year prices, over the GDP deflator).

    Current prices are in units of the numeraire. Base-year prices are the
    calibrated year's: final demand there is what ``price_indices`` calls the
    Laspeyres volume, and GDP is the expenditure side of ``real_gdp``.
    ``pct_base_prices`` is the change in volume. ``unit`` multiplies every level
    and change: a SAM in millions and ``unit=1e-3`` gives billions. ``label``,
    the unit's name, is kept in ``attrs["unit"]``; the package does not know
    the SAM's currency.
    """
    from .dynamics import PathResult

    if not (float(unit) > 0 and np.isfinite(float(unit))):
        raise ValueError(f"unit must be a positive number, got {unit!r}")
    if isinstance(reference, PathResult) != isinstance(alternative, PathResult):
        raise TypeError("reference and alternative must both be periods or both be paths")
    if isinstance(reference, PathResult):
        shared = [y for y in reference.years if y in alternative.years]
        chosen = shared if years is None else list(years)
        missing = [y for y in chosen if y not in shared]
        if missing:
            raise ValueError(f"years {missing} are not in both paths")
        if not chosen:
            raise ValueError("the two paths share no year")
        out = pd.concat({y: _money_table(reference[y], alternative[y], unit) for y in chosen},
                        names=["year"])
    else:
        if years is not None:
            raise ValueError("years applies to paths, not to two periods")
        if reference.econ.hhs != alternative.econ.hhs or \
                reference.econ.factors != alternative.econ.factors:
            raise ValueError("the two periods come from economies with different accounts")
        out = _money_table(reference, alternative, unit)
    out.attrs["unit"] = label
    return out
