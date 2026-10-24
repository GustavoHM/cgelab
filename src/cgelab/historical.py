"""Limited historical closure: observed targets, endogenous model shifters.

The period equations stay intact. A historical calibration makes observed
variables targets and solves the technology or preference multipliers that
reproduce them. Recursive years are solved in sequence because capital carries
from the kept solution of one year to the next.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, root

from .calibrate import Economy
from .closure import Closure
from .dynamics import Growth, PathResult, _PathState, run_path
from .scenario import Scenario

__all__ = ["Historical", "calibrate_historical"]

_KINDS = ("growth", "index", "level")


@dataclass
class Historical:
    """A path calibrated to observed targets by technology and preferences."""

    years: list
    path: PathResult
    target: pd.DataFrame
    achieved: pd.DataFrame
    productivity: pd.DataFrame
    multiplier: pd.DataFrame
    preferences: dict[str, pd.DataFrame]
    econ: Economy = field(repr=False)
    closure: Closure = field(repr=False)
    growth: Growth = field(repr=False)
    economy: Callable | None = field(default=None, repr=False)
    base_scenario: Scenario | Callable | None = field(default=None, repr=False)

    def scenario(self, policy: Scenario | Callable | None = None) -> Callable:
        """The scenario at calibrated productivity, with an optional policy."""
        index = {y: i for i, y in enumerate(self.years)}
        mult = self.multiplier.to_numpy()
        base = self.base_scenario if policy is None else policy

        def f(year, t):
            if year not in index:
                raise ValueError(f"year {year!r} is not in the historical calibration")
            sc = _scenario_at(base, year, t)
            own = (np.ones(self.econ.n_act) if sc.productivity is None
                   else np.asarray(sc.productivity, dtype=float))
            return sc.with_(productivity=own * mult[index[year]])

        return f

    def _preferences_at(self, econ, year):
        if year not in self.years:
            raise ValueError(f"year {year!r} is not in the historical calibration")
        i = self.years.index(year)
        changes = {}
        for household, multipliers in self.preferences.items():
            h = econ.hhs.index(household)
            budget = np.asarray(econ.hh_budget, dtype=float).copy()
            weights = budget[h] * multipliers.iloc[i].to_numpy(dtype=float)
            budget[h] = weights / weights.sum()
            changes["hh_budget"] = budget
        return dataclasses.replace(econ, **changes) if changes else econ

    def run(self, policy: Scenario | Callable | None = None, *,
            raise_on_fail: bool = True) -> PathResult:
        """Run a policy while holding the calibrated technology and preferences."""
        def at_economy(year, t):
            base = self.economy(year, t) if self.economy is not None else self.econ
            return self._preferences_at(base, year)

        return run_path(self.econ, self.closure, years=self.years,
                        scenario=self.scenario(policy), growth=self.growth,
                        economy=at_economy, raise_on_fail=raise_on_fail)


def _scenario_at(scenario, year, t):
    return scenario(year, t) if callable(scenario) else (scenario or Scenario())


def _growth_level(growth, t, n_act):
    if growth.productivity == "factor_augmenting":
        return np.ones(n_act)
    return np.full(n_act, (1.0 + growth.tfp) ** t)


def _cover(index, years, kind, name):
    need = years[1:] if kind == "growth" else years
    missing = [y for y in need if y not in index]
    if missing:
        raise ValueError(f"{name} has no value for {missing}; with kind={kind!r} it needs "
                         f"{'every year after the first' if kind == 'growth' else 'every year'}")


def _target(series, years, t, kind, first, prev):
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
        raise ValueError(f"the target for {year!r} is {out!r}; historical targets must be positive")
    return out


def _pattern(econ, pattern):
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
        raise ValueError("pattern must be finite and not all zero")
    return out


def _scalar_root(solve, measured, target, x0, year, tol):
    """One log multiplier that makes a scalar measure hit its target."""
    log_target = np.log(target)

    def gap(x):
        res, _ = solve(x, raise_on_fail=False)
        if not res.converged:
            return np.nan
        value = measured(res)
        return np.log(value) - log_target if value > 0 else -np.inf

    g0 = gap(x0)
    if not np.isfinite(g0):
        raise RuntimeError(f"{year!r}: the period does not solve at last year's shifters")
    if abs(g0) <= tol:
        return x0
    step = -0.25 if g0 > 0 else 0.25
    a, ga = x0, g0
    for _ in range(12):
        b = a + step
        gb = gap(b)
        if not np.isfinite(gb):
            break
        if np.sign(gb) != np.sign(g0) or gb == 0.0:
            return brentq(gap, min(a, b), max(a, b), xtol=1e-15,
                          rtol=4 * np.finfo(float).eps, maxiter=200)
        a, ga = b, gb
        step *= 2
    raise ValueError(
        f"{year!r}: no shifter reaches the target {target:.6g}; the closest was "
        f"{target * np.exp(ga):.6g}, at a multiplier of {np.exp(a):.4g}.")


def _vector_root(solve, measured, target, x0, year, tol, names):
    """One log multiplier per target, solved jointly."""
    log_target = np.log(target)

    def gap(x):
        res, _ = solve(x, raise_on_fail=False)
        value = measured(res)
        if not res.converged or (value <= 0).any():
            return np.full(len(names), 1e3)
        return np.log(value) - log_target

    sol = root(gap, np.asarray(x0, dtype=float), method="hybr", tol=1e-14)
    worst = float(np.max(np.abs(gap(sol.x))))
    if worst > max(tol, 1e-10):
        miss = dict(zip(names, np.round(gap(sol.x), 6)))
        raise ValueError(f"{year!r}: no shifters reach the targets; log gaps left: "
                         f"{miss}. {sol.message}")
    return sol.x


def _specification(econ, targets, shifters, pattern, preference_reference):
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a nonempty mapping")
    if set(targets) != set(shifters):
        raise ValueError("targets and shifters must name the same entries")
    names = list(targets)
    specs = [shifters[name] for name in names]
    if len(set(specs)) != len(specs):
        raise ValueError("each target needs a distinct shifter")
    if pattern is not None and any(spec != ("productivity",) for spec in specs):
        raise ValueError("pattern applies only to one aggregate productivity shifter")
    reference = dict(preference_reference or {})
    parsed = []
    for name, spec in zip(names, specs):
        if not isinstance(spec, tuple):
            raise TypeError(f"shifter {name!r} must be a tuple")
        if spec[0] == "productivity" and len(spec) in (1, 2):
            if len(spec) == 2 and spec[1] not in econ.acts:
                raise KeyError(f"unknown activity in shifter {name!r}: {spec[1]!r}")
        elif spec[0] == "preference" and len(spec) == 3:
            _, household, commodity = spec
            if household not in econ.hhs or commodity not in econ.coms:
                raise KeyError(f"unknown household or commodity in shifter {name!r}: {spec}")
            if household not in reference:
                raise ValueError(f"preference_reference has no reference commodity for {household!r}")
            if reference[household] not in econ.coms:
                raise KeyError(f"unknown reference commodity for {household!r}: "
                               f"{reference[household]!r}")
            if reference[household] == commodity:
                raise ValueError(f"shifter {name!r} must not move the reference commodity")
        else:
            raise ValueError(f"unknown shifter {name!r}: {spec}")
        parsed.append(spec)
    return names, parsed


def calibrate_historical(econ: Economy, closure: Closure | None = None, *, years: Sequence,
                         targets: Mapping, shifters: Mapping, kind: str = "level",
                         growth: Growth | None = None, scenario: Scenario | Callable | None = None,
                         economy: Callable | None = None, pattern=None,
                         preference_reference: Mapping[str, str] | None = None,
                         tol: float = 1e-12) -> Historical:
    """Calibrate technology and preference multipliers to observed targets.

    ``targets`` maps each name to ``(observed Series, measure(result))``.
    ``shifters`` gives the corresponding ``("productivity",)``,
    ``("productivity", activity)`` or ``("preference", household, commodity)``.
    One log multiplier per target is solved jointly each year. Preferences are
    normalised by household, with the multiplier of ``preference_reference``'s
    commodity fixed at one. ``kind`` has the baseline meaning: levels, an index
    normalised by its first year, or growth rates chained from the first year.
    """
    closure = closure or Closure()
    growth = growth or Growth()
    years = list(years)
    if not years:
        raise ValueError("calibrate_historical needs at least one year")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, not {kind!r}")
    names, specs = _specification(econ, targets, shifters, pattern, preference_reference)
    series = {name: pd.Series(targets[name][0], dtype=float) for name in names}
    measures = {name: targets[name][1] for name in names}
    for name in names:
        if len(targets[name]) != 2 or not callable(measures[name]):
            raise TypeError(f"target {name!r} must be (observed values, measure)")
        _cover(series[name].index, years, kind, name)
    expo = _pattern(econ, pattern)
    n_act = econ.n_act
    state = _PathState(econ, closure, growth)
    first = prev_target = None
    x_prev = np.zeros(len(names))
    mults, prefs_rows, target_rows, achieved_rows, prods = [], [], [], [], []

    for t, year in enumerate(years):
        e_t = state.economy_at(economy, year, t)
        sc = _scenario_at(scenario, year, t)
        own = np.ones(n_act) if sc.productivity is None else np.asarray(sc.productivity, float)

        def multipliers(x):
            productivity = np.ones(n_act)
            preferences = {}
            for value, name, spec in zip(np.atleast_1d(x), names, specs):
                if spec[0] == "productivity":
                    if len(spec) == 1:
                        productivity *= np.exp(value * expo)
                    else:
                        productivity[econ.activity(spec[1])] *= np.exp(value)
                else:
                    _, household, commodity = spec
                    preferences.setdefault(household, np.ones(econ.n_com))
                    preferences[household][econ.coms.index(commodity)] *= np.exp(value)
            return productivity, preferences

        def adjusted_economy(preferences):
            changes = {}
            for household, values in preferences.items():
                h = econ.hhs.index(household)
                budget = np.asarray(e_t.hh_budget, dtype=float).copy()
                weights = budget[h] * values
                budget[h] = weights / weights.sum()
                changes["hh_budget"] = budget
            return dataclasses.replace(e_t, **changes) if changes else e_t

        def solve(x, warm=None, raise_on_fail=True):
            productivity, preferences = multipliers(x)
            return state.solve(adjusted_economy(preferences),
                               sc.with_(productivity=own * productivity), t,
                               warm=warm, raise_on_fail=raise_on_fail)

        def measured(res):
            return np.array([float(measures[name](res)) for name in names])

        calibrated = not (t == 0 and kind != "level")
        if not calibrated:
            x = np.zeros(len(names))
            res, sc_full = solve(x)
            first = measured(res)
            target = first
        else:
            target = np.array([_target(series[name], years, t, kind,
                                       None if first is None else first[j],
                                       None if prev_target is None else prev_target[j])
                               for j, name in enumerate(names)])
            if len(names) == 1:
                x = np.array([_scalar_root(solve, lambda res: measured(res)[0], target[0],
                                           x_prev[0], year, tol)])
            else:
                x = _vector_root(solve, measured, target, x_prev, year, tol, names)
            res, sc_full = solve(x)
        productivity, preferences = multipliers(x)
        reached = measured(res)
        state.advance(adjusted_economy(preferences), res, sc_full)
        mults.append(productivity)
        prefs_rows.append(preferences)
        target_rows.append(target)
        achieved_rows.append(reached)
        prods.append(np.asarray(sc_full.productivity, float) / _growth_level(growth, t, n_act))
        prev_target = target
        x_prev = np.atleast_1d(x)

    productivity = pd.DataFrame(np.array(prods), index=years, columns=list(econ.acts))
    multiplier = pd.DataFrame(np.array(mults), index=years, columns=list(econ.acts))
    preferences = {}
    for household in {h for row in prefs_rows for h in row}:
        preferences[household] = pd.DataFrame(
            [row.get(household, np.ones(econ.n_com)) for row in prefs_rows],
            index=years, columns=list(econ.coms))
    return Historical(years=years, path=state.result(years),
                      target=pd.DataFrame(target_rows, index=years, columns=names),
                      achieved=pd.DataFrame(achieved_rows, index=years, columns=names),
                      productivity=productivity, multiplier=multiplier,
                      preferences=preferences, econ=econ, closure=closure, growth=growth,
                      economy=economy, base_scenario=scenario)
