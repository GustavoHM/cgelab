"""Minimum-distance matching of model outcomes to external targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .sensitivity import _as_dict, spec_sweep

__all__ = ["MatchResult", "match_targets"]

_ERRORS = (ValueError, RuntimeError, KeyError, TypeError, np.linalg.LinAlgError)


@dataclass
class MatchResult:
    """The best evaluation and the optimiser's evidence, including failure."""

    parameters: pd.Series
    achieved: pd.Series
    target: pd.Series
    residuals: pd.Series
    distance: float
    success: bool
    message: str
    nit: int
    nfev: int
    evaluations: pd.DataFrame = field(repr=False)
    spec: object = field(default=None, repr=False)

    def __repr__(self):
        return (f"MatchResult(success={self.success}, distance={self.distance:.6g}, "
                f"nit={self.nit}, nfev={self.nfev})")


def _bounds(parameters):
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a nonempty mapping of names to (low, high)")
    names, bounds = list(parameters), []
    for name in names:
        try:
            low, high = map(float, parameters[name])
        except (TypeError, ValueError) as exc:
            raise TypeError(f"parameter {name!r} must be a (low, high) pair") from exc
        if not np.isfinite([low, high]).all() or not low < high:
            raise ValueError(f"parameter {name!r} needs finite low < high, got {(low, high)}")
        bounds.append((low, high))
    return names, bounds


def _pseudoinverse(covariance, n):
    if covariance is None:
        return np.eye(n)
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 1:
        if len(cov) != n:
            raise ValueError(f"covariance diagonal has length {len(cov)}, expected {n}")
        if (cov <= 0).any():
            raise ValueError("covariance diagonal entries must be positive")
        cov = np.diag(cov)
    if cov.shape != (n, n):
        raise ValueError(f"covariance has shape {cov.shape}, expected {(n, n)}")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return np.linalg.pinv(cov, rcond=1e-12, hermitian=True)


def _coefficients(value, names):
    if len(names) == 1 and isinstance(value, (int, float, np.floating, np.integer)):
        return {names[0]: float(value)}
    values = _as_dict(value)
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"measure did not return targets {missing}")
    return {name: float(values[name]) for name in names}


def _match(run, targets, parameters, *, covariance=None, options=None):
    """Minimise weighted squared residuals of ``run`` against named targets."""
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a nonempty mapping of names to numbers")
    target_names = list(targets)
    target = np.array([float(targets[name]) for name in target_names])
    if not np.isfinite(target).all():
        raise ValueError("targets must contain only finite values")
    names, bounds = _bounds(parameters)
    pinv = _pseudoinverse(covariance, len(target_names))
    midpoint = np.array([(a + b) / 2 for a, b in bounds])
    evaluations = []

    def evaluate(x):
        point = dict(zip(names, map(float, x)))
        try:
            achieved = _coefficients(run(point), target_names)
            residual = np.array([achieved[name] for name in target_names]) - target
            distance = float(max(0.0, residual @ pinv @ residual))
            if not np.isfinite(distance):
                raise ValueError("the weighted distance is not finite")
            error = ""
        except _ERRORS as exc:
            achieved = {name: np.nan for name in target_names}
            distance = 1e100
            error = f"{type(exc).__name__}: {exc}"
        evaluations.append({**point, "distance": distance, "error": error,
                            **{f"achieved[{name}]": achieved[name] for name in target_names}})
        return distance

    options = dict(options or {})
    options.setdefault("maxiter", 200)
    options.setdefault("ftol", 1e-12)
    options.setdefault("gtol", 1e-8)
    first = evaluate(midpoint)
    if first >= 1e100:
        raise RuntimeError(f"the initial parameter midpoint failed: {evaluations[-1]['error']}")
    opt = minimize(evaluate, midpoint, method="L-BFGS-B", bounds=bounds, options=options)
    best = min(evaluations, key=lambda row: row["distance"])
    opt.success = bool(opt.success or best["distance"] <= options["ftol"])
    achieved = {name: best[f"achieved[{name}]"] for name in target_names}
    residuals = {name: achieved[name] - float(targets[name]) for name in target_names}
    return opt, best, achieved, residuals, pd.DataFrame(evaluations)


def match_targets(sam, spec, *, targets: Mapping, parameters: Mapping,
                  measure: Callable, covariance=None, scenario=None, closure=None,
                  calibrate_kwargs: Mapping | None = None,
                  options: Mapping | None = None) -> MatchResult:
    """Fit ModelSpec fields by weighted minimum distance to external coefficients.

    ``targets`` maps coefficient names to their published values. ``parameters``
    maps fields accepted by :func:`spec_sweep` to finite ``(low, high)`` bounds.
    ``measure(reference, alternative)`` returns those coefficients. ``covariance``
    may be a full matrix or a diagonal of variances; absent, it is the identity.
    A singular positive-semidefinite covariance is handled by its pseudo-inverse.
    ``options`` goes to ``scipy.optimize.minimize`` with L-BFGS-B.

    This is a deterministic match to external numbers, not an econometric
    estimator. The result reports optimiser failure rather than hiding it.
    """
    opt, best, achieved, residuals, evaluations = _match(
        spec_sweep(sam, spec, measure=measure, closure=closure, scenario=scenario,
                   calibrate_kwargs=calibrate_kwargs),
        targets, parameters, covariance=covariance, options=options)
    from .sensitivity import _replace_spec
    fitted_spec = _replace_spec(spec, {name: best[name] for name in parameters})
    target = pd.Series({name: float(targets[name]) for name in targets})
    return MatchResult(parameters=pd.Series({name: best[name] for name in parameters}),
                       achieved=pd.Series(achieved), target=target,
                       residuals=pd.Series(residuals), distance=float(best["distance"]),
                       success=bool(opt.success or best["distance"] <= (options or {}).get("ftol", 1e-12)),
                       message=str(opt.message), nit=int(opt.nit),
                       nfev=len(evaluations), evaluations=evaluations,
                       spec=fitted_spec)
