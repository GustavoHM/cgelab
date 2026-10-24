"""Calibrating a recursive baseline to a path of real GDP.

A baseline built from assumed rates -- ``Growth.tfp``, ``Growth.labour`` -- makes
GDP whatever those rates give. The applied practice, from the historical
simulations of MONASH to the baselines of LINKAGE, turns this round: real GDP in
each year, observed or projected, is given, and productivity is the unknown that
reproduces it. A policy run then holds productivity at the calibrated values and
lets GDP move; the gap between the two paths is the policy's effect.

The years are calibrated in sequence, because each year's capital stock comes
from the investment of the years before: one scalar root a year, or one root
per activity when sectoral value added is the target. Nothing in the period
equations changes. The calibrated productivity enters through
``Scenario.productivity``, on top of what the growth block and the scenario
already bring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .calibrate import Economy
from .closure import Closure
from .dynamics import Growth, PathResult, _PathState, run_path
from .historical import _scalar_root, _vector_root
from .results import real_gdp, real_value_added
from .scenario import Scenario
from .solve import PeriodResult

__all__ = ["Baseline", "calibrate_baseline"]

_KINDS = ("growth", "index", "level")
_MEASURES = ("expenditure", "production")


@dataclass
class Baseline:
    """A calibrated baseline: the productivity that reproduces a GDP path.

    ``productivity`` is the multiplier each year passed to the solver as
    ``Scenario.productivity``: the calibrated part times whatever the
    baseline's scenario already set. The growth block's own productivity,
    ``Growth.tfp``, comes on top of it, as in :func:`run_path`. ``factor`` is
    the calibrated part alone: one number a year, ``Λ_t``, or with sectoral
    targets one column per calibrated activity. ``target`` and ``achieved`` are
    the real GDP (or value added) asked for and the one the path reached, in the
    SAM's units at base-year prices.
    """

    years: list
    productivity: pd.DataFrame
    factor: pd.Series | pd.DataFrame
    path: PathResult
    target: pd.Series | pd.DataFrame
    achieved: pd.Series | pd.DataFrame
    measure: str = "expenditure"
    econ: Economy = field(default=None, repr=False)
    closure: Closure = field(default=None, repr=False)
    growth: Growth = field(default=None, repr=False)
    economy: Callable | None = field(default=None, repr=False)
    base_scenario: Scenario | Callable | None = field(default=None, repr=False)
    #: The calibrated multiplier alone, ``(n_years, n_act)``.
    multiplier: pd.DataFrame = field(default=None, repr=False)

    def scenario(self, policy: Scenario | Callable | None = None) -> Callable:
        """``f(year, t) -> Scenario`` for :func:`run_path`, at the calibrated productivity.

        With no policy it is the baseline's own scenario, and running it
        reproduces the baseline. A policy is the whole scenario for the year, a
        :class:`Scenario` or ``f(year, t) -> Scenario``, built from the
        baseline's scenario as any policy is built from its reference; a
        productivity it sets multiplies the calibrated one. Years outside the
        baseline are refused.
        """
        index = {y: i for i, y in enumerate(self.years)}
        mult = self.multiplier.to_numpy()
        base = self.base_scenario if policy is None else policy

        def f(year, t):
            if year not in index:
                raise ValueError(f"year {year!r} is not in the calibrated baseline "
                                 f"({self.years[0]!r} to {self.years[-1]!r})")
            sc = _scenario_at(base, year, t)
            own = (np.ones(self.econ.n_act) if sc.productivity is None
                   else np.asarray(sc.productivity, dtype=float))
            return sc.with_(productivity=own * mult[index[year]])

        return f

    def run(self, policy: Scenario | Callable | None = None, *,
            raise_on_fail: bool = True) -> PathResult:
        """The path under ``policy`` at the calibrated productivity: the policy run."""
        return run_path(self.econ, self.closure, years=self.years,
                        scenario=self.scenario(policy), growth=self.growth,
                        economy=self.economy, raise_on_fail=raise_on_fail)


def calibrate_baseline(econ: Economy, closure: Closure | None = None, *, years: Sequence,
                       gdp: pd.Series | Mapping | None = None, kind: str = "growth",
                       growth: Growth | None = None,
                       scenario: Scenario | Callable | None = None,
                       economy: Callable | None = None,
                       pattern: pd.Series | Mapping | Sequence | None = None,
                       sectors: pd.DataFrame | None = None,
                       measure: str = "expenditure",
                       tol: float = 1e-12) -> Baseline:
    """Find the productivity path that makes a recursive baseline hit a GDP path.

    Parameters
    ----------
    gdp : Series
        Real GDP by year. With ``kind="growth"``, annual growth rates (0.03 for
        3%): the first year is solved as :func:`run_path` would solve it, and
        the rates chain from its GDP, so a rate for the first year is not read.
        With ``kind="index"``, an index of real GDP, normalised by its first
        year, which is also left uncalibrated. With ``kind="level"``, levels in
        the SAM's units at base-year prices, and every year is calibrated, the
        first one too.
    pattern : Series or array, optional
        By activity. The year's unknown is one number ``Λ_t``, and activity
        ``a``'s productivity is multiplied by ``Λ_t ** pattern[a]``: with no
        pattern, the same in every activity. A pattern says that farming gains
        twice what services gain (2 and 1) while the path fixes how much in
        all. Activities a mapping leaves out get 0, no calibrated change.
    sectors : DataFrame, optional
        Real value added by year (rows) and activity (columns), read with
        ``kind`` as ``gdp`` is: one unknown per activity and year, solved
        together. Activities not among the columns keep the productivity the
        growth block and the scenario give them. Not with ``gdp``.
    measure : str
        For ``gdp``: ``"expenditure"``, the expenditure side of
        :func:`real_gdp`, or ``"production"``, the sum of
        :func:`real_value_added`. Sectoral targets are always the latter, by
        activity.
    growth, scenario, economy
        As in :func:`run_path`; the calibrated productivity multiplies
        everything they set.
    tol : float
        On the log of the target: each year is solved until
        ``|log(achieved / target)|`` is below it.
    """
    closure = closure or Closure()
    growth = growth or Growth()
    years = list(years)
    if not years:
        raise ValueError("calibrate_baseline needs at least one year")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, not {kind!r}")
    if measure not in _MEASURES:
        raise ValueError(f"measure must be one of {_MEASURES}, not {measure!r}")
    if (gdp is None) == (sectors is None):
        raise ValueError("give the GDP path (gdp=) or sectoral value added (sectors=), "
                         "one of them")
    if sectors is not None and pattern is not None:
        raise ValueError("pattern shares one unknown among activities; with sectors= each "
                         "activity has its own")
    n_act = econ.n_act

    if gdp is not None:
        series = pd.Series(gdp, dtype=float)
        _cover(series.index, years, kind, "gdp")
        expo = _pattern(econ, pattern)
        cols = list(econ.acts)
    else:
        sectors = pd.DataFrame(sectors, dtype=float)
        unknown = [a for a in sectors.columns if a not in econ.acts]
        if unknown:
            raise ValueError(f"sectors names activities that are not in the economy: {unknown}")
        if sectors.columns.duplicated().any():
            raise ValueError("sectors names an activity twice")
        _cover(sectors.index, years, kind, "sectors")
        cols = list(sectors.columns)
        idx = np.array([econ.activity(a) for a in cols])

    state = _PathState(econ, closure, growth)
    mults, factors, targets, achieved, prods = [], [], [], [], []
    first = None                      # the first year's measure, for growth and index
    prev_target = None
    x_prev = 0.0 if gdp is not None else np.zeros(len(cols))

    for t, year in enumerate(years):
        e_t = state.economy_at(economy, year, t)
        sc = _scenario_at(scenario, year, t)
        own = (np.ones(n_act) if sc.productivity is None
               else np.asarray(sc.productivity, dtype=float))
        calibrated = not (t == 0 and kind != "level")

        if gdp is not None:
            def multiplier(x):
                return np.exp(x * expo)

            def measured(res):
                return _measure(res, measure)
        else:
            def multiplier(x):
                m = np.ones(n_act)
                m[idx] = np.exp(x)
                return m

            def measured(res):
                return real_value_added(res).to_numpy()[idx]

        def solve(x, warm=None, raise_on_fail=True):
            return state.solve(e_t, sc.with_(productivity=own * multiplier(x)), t,
                               warm=warm, raise_on_fail=raise_on_fail)

        if not calibrated:
            x = 0.0 * x_prev
            res, sc_full = solve(x)
            first = measured(res)
            target = first
        else:
            if gdp is not None:
                target = _target(series, years, t, kind, first, prev_target)
                x = _scalar_root(solve, measured, target, x_prev, year, tol)
            else:
                target = np.array([_target(sectors[a], years, t, kind,
                                           None if first is None else first[j],
                                           None if prev_target is None else prev_target[j])
                                   for j, a in enumerate(cols)])
                x = _vector_root(solve, measured, target, x_prev, year, tol, cols)
            # The kept solution is solved as run_path would solve it, warm from
            # the year before, so that rerunning the baseline gives it back.
            res, sc_full = solve(x)
        reached = measured(res)
        state.advance(e_t, res, sc_full)
        mults.append(multiplier(x))
        factors.append(np.exp(x))
        targets.append(target)
        achieved.append(reached)
        prods.append(np.asarray(sc_full.productivity, dtype=float) / _growth_level(growth, t, n_act))
        prev_target = target
        x_prev = x

    path = state.result(years)
    frame = lambda rows, c: pd.DataFrame(np.array(rows), index=years, columns=c)  # noqa: E731
    if gdp is not None:
        factor = pd.Series(np.array(factors, dtype=float), index=years, name="factor")
        tgt = pd.Series(np.array(targets, dtype=float), index=years, name="target")
        got = pd.Series(np.array(achieved, dtype=float), index=years, name="achieved")
    else:
        factor, tgt, got = frame(factors, cols), frame(targets, cols), frame(achieved, cols)
    return Baseline(years=years, productivity=frame(prods, list(econ.acts)), factor=factor,
                    path=path, target=tgt, achieved=got,
                    measure=measure if gdp is not None else "production", econ=econ,
                    closure=closure, growth=growth, economy=economy, base_scenario=scenario,
                    multiplier=frame(mults, list(econ.acts)))


def _scenario_at(scenario, year, t) -> Scenario:
    return scenario(year, t) if callable(scenario) else (scenario or Scenario())


def _growth_level(growth: Growth, t: int, n_act: int) -> np.ndarray:
    """The Hicks-neutral level the growth block adds in year ``t``, to take back out."""
    if growth.productivity == "factor_augmenting":
        return np.ones(n_act)
    return np.full(n_act, (1.0 + growth.tfp) ** t)


def _measure(res: PeriodResult, measure: str) -> float:
    if measure == "expenditure":
        return real_gdp(res)["expenditure"]
    return float(real_value_added(res).sum())


def _cover(index, years, kind, name):
    need = years[1:] if kind == "growth" else years
    missing = [y for y in need if y not in index]
    if missing:
        raise ValueError(f"{name} has no value for {missing}; with kind={kind!r} it needs "
                         f"{'every year after the first' if kind == 'growth' else 'every year'}")


def _target(series: pd.Series, years, t, kind, first, prev):
    year = years[t]
    value = float(series[year])
    if not np.isfinite(value):
        raise ValueError(f"the target for {year!r} is not a number: {value!r}")
    if kind == "growth":
        out = prev * (1.0 + value)
    elif kind == "index":
        out = first * value / float(series[years[0]])
    else:
        out = value
    if not out > 0:
        raise ValueError(f"the target for {year!r} is {out!r}; real GDP and value added "
                         f"must be positive")
    return out


def _pattern(econ: Economy, pattern) -> np.ndarray:
    if pattern is None:
        return np.ones(econ.n_act)
    if isinstance(pattern, (Mapping, pd.Series)):
        items = dict(pattern)
        unknown = [a for a in items if a not in econ.acts]
        if unknown:
            raise ValueError(f"pattern names activities that are not in the economy: {unknown}")
        out = np.array([float(items.get(a, 0.0)) for a in econ.acts])
    else:
        out = np.asarray(pattern, dtype=float)
        if out.shape != (econ.n_act,):
            raise ValueError(f"pattern has {out.size} values for {econ.n_act} activities")
    if not np.isfinite(out).all() or not out.any():
        raise ValueError("pattern must be finite and not all zero: it is how the year's "
                         "unknown reaches the activities")
    return out
