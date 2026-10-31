"""Tests a calibrated CGE must pass before any result from it is believable.

These are not optional extras. A model that fails the base-year reproduction is
not calibrated; one that fails homogeneity has a nominal magnitude leaking into a
real allocation; one with a Walras residual is losing or manufacturing income
somewhere, and the size of the leak bounds nothing about the size of the error it
causes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .calibrate import Economy
from .closure import Closure
from .scenario import Scenario
from .solve import PeriodResult, solve_period

__all__ = ["base_year_reproduction", "homogeneity_test", "walras_check", "report",
           "InvestmentBounds", "investment_bounds"]


def base_year_reproduction(econ: Economy, closure: Closure | None = None,
                           tol: float = 1e-7) -> dict:
    """With no shock, the model must return the SAM.

    Every price at one, every quantity at its base-year value. This is the test
    that catches a units error in the calibration -- a purchaser price treated as
    a basic price, a tax levied on the wrong base -- and it catches it before the
    error has been absorbed into a result.
    """
    res = solve_period(econ, closure or Closure(), raise_on_fail=False)
    checks = {
        "activity_output": _rel(res.X, econ.X0),
        "commodity_output": _rel(res.Q, econ.Q0),
        "activity_price": _rel(res.pa, np.ones_like(res.pa)),
        "consumer_price": _rel(res.pq, 1.0 + econ.ptax),
        "factor_price": _rel(res.factor_prices, np.ones_like(res.factor_prices)),
        "factor_demand": _rel(res.factor_demand, econ.fac_supply),
        "household_income": _rel(res.hh_income, econ.hh_income0 if econ.hh_gross0 is None
                                 else econ.hh_gross0),
        "exchange_rate": abs(res.exchange_rate - 1.0),
        "imports": _rel(res.imports, econ.imp0),
        "exports": _rel(res.exports, econ.exp0),
    }
    worst = max(checks.values())
    return dict(ok=bool(worst < tol and abs(res.walras) < tol), worst=float(worst),
                walras=float(res.walras), checks=checks, result=res, tolerance=tol)


def homogeneity_test(econ: Economy, closure: Closure | None = None,
                     factor: float = 2.5, tol: float = 1e-7) -> dict:
    """Doubling the numeraire must leave every real magnitude untouched.

    Zero-degree homogeneity in prices is a property of the model, not of the
    solver, so a failure means a nominal magnitude denominated in the numeraire
    was held fixed while the price level moved -- foreign saving, the
    trade-balance target, or a specific tax that enters additively rather than ad
    valorem. Each of those failures shifts a real allocation and none of them is
    visible in the output.
    """
    closure = closure or Closure()
    base = solve_period(econ, closure, raise_on_fail=False)
    # An exogenous exchange rate is a nominal magnitude too: holding it while the
    # numeraire scales would move the real exchange rate, and the test would
    # fail for a reason that is the test's.
    changes = {"numeraire": closure.numeraire * factor}
    if not closure.solves_exchange_rate:
        changes["exchange_rate"] = closure.exchange_rate * factor
    scaled = solve_period(econ, Closure(**{**closure.__dict__, **changes}), raise_on_fail=False)
    checks = {
        "activity_output": _rel(scaled.X, base.X),
        "commodity_output": _rel(scaled.Q, base.Q),
        "household_quantities": _rel(scaled.hh_quantity, base.hh_quantity),
        "real_absorption": abs(scaled.real["hh_real"] / base.real["hh_real"] - 1.0),
        "relative_prices": _rel(scaled.pq / scaled.pq[0], base.pq / base.pq[0]),
        "exchange_rate": abs(scaled.exchange_rate / base.exchange_rate / factor - 1.0),
    }
    worst = max(checks.values())
    return dict(ok=bool(worst < tol), worst=float(worst), checks=checks,
                factor=factor, tolerance=tol)


def walras_check(result: PeriodResult, tol: float = 1e-8) -> dict:
    """The market the solver was NOT told to clear must clear anyway."""
    return dict(ok=bool(abs(result.walras) < tol), residual=float(result.walras),
                factor=result.econ.spec.numeraire_factor, tolerance=tol)


def report(econ: Economy, closure: Closure | None = None) -> str:
    """Run every check and render a short verdict, for a log or a referee appendix."""
    base = base_year_reproduction(econ, closure)
    homo = homogeneity_test(econ, closure)
    lines = [f"cgelab diagnostics -- {econ!r}",
             f"  base-year reproduction : {'PASS' if base['ok'] else 'FAIL'} "
             f"(worst {base['worst']:.2e})"]
    for name, v in sorted(base["checks"].items(), key=lambda kv: -kv[1])[:4]:
        lines.append(f"      {name:<22} {v:.2e}")
    lines.append(f"  Walras residual        : {base['walras']:.2e} "
                 f"[{econ.spec.numeraire_factor}]")
    lines.append(f"  price homogeneity      : {'PASS' if homo['ok'] else 'FAIL'} "
                 f"(worst {homo['worst']:.2e})")
    if econ.diagnostics.get("negative_value_added"):
        acts = ", ".join(econ.diagnostics["negative_value_added"])
        lines.append(f"  negative value added   : {acts} "
                     f"-> {econ.diagnostics['negative_value_added_treatment']}")
    return "\n".join(lines)


@dataclass
class InvestmentBounds:
    """The ``investment_level`` values a ``fixed_real`` period solves at, bracketed.

    ``low`` and ``high`` solve; a level beyond either, by more than the
    tolerance the search stopped at, did not. A bound the search never crossed
    is 0 below and ``inf`` above. ``below`` and ``above`` hold what the first
    failing level on that side raised or reported, so the binding constraint is
    named, not only located.
    """

    low: float
    high: float
    reference: float
    below: str | None = None
    above: str | None = None


def investment_bounds(econ: Economy, closure: Closure | None = None,
                      scenario: Scenario | None = None, *, rtol: float = 1e-3,
                      doublings: int = 12) -> InvestmentBounds:
    """Bracket the real investment a ``fixed_real`` closure can finance, by bisection.

    The search starts from the scenario's ``investment_level`` (the base-year
    investment when it has none), which must solve. Upward, the level doubles
    until a period fails, at most ``doublings`` times; downward, zero is tried
    first. Each bracket is then halved until it is narrower than ``rtol`` times
    the reference level. The closure is solved with ``investment="fixed_real"``
    whatever it says, since that is the closure the question is about.

    Every probe is a full ``solve_period``, so this costs a few dozen solves.
    The bounds belong to these prices and this scenario: another shock moves them.
    """
    closure = dataclasses.replace(closure or Closure(), investment="fixed_real")
    scenario = scenario or Scenario()
    reference = (float(econ.inv_dem.sum()) if scenario.investment_level is None
                 else float(scenario.investment_level))

    def probe(level):
        sc = dataclasses.replace(scenario, investment_level=float(level))
        try:
            res = solve_period(econ, closure, sc, raise_on_fail=False)
        except (ValueError, RuntimeError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if not res.converged:
            return False, f"residual {res.residual:.3e} above tolerance at the best start"
        return True, None

    ok, why = probe(reference)
    if not ok:
        raise ValueError(f"investment_level={reference:.6g} does not solve, so there is no "
                         f"bracket to start from. It failed with: {why}")
    step = rtol * max(abs(reference), 1e-12)

    def bisect(good, bad, why):
        while abs(bad - good) > step:
            mid = 0.5 * (good + bad)
            ok, reason = probe(mid)
            if ok:
                good = mid
            else:
                bad, why = mid, reason
        return good, why

    high, above = np.inf, None
    level = reference
    for _ in range(doublings):
        trial = 2.0 * level if level > 0 else 1.0
        ok, reason = probe(trial)
        if not ok:
            high, above = bisect(level, trial, reason)
            break
        level = trial
    low, below = 0.0, None
    if reference > 0:
        ok, reason = probe(0.0)
        if not ok:
            low, below = bisect(reference, 0.0, reason)
    return InvestmentBounds(low=float(low), high=float(high), reference=reference,
                            below=below, above=above)


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(b), 1e-9)
    return float(np.max(np.abs(a - b) / denom))
