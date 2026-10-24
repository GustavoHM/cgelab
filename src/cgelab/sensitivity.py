"""Systematic sensitivity: a result as a distribution over uncertain parameters.

Every applied study sweeps its elasticities with a loop of its own. This module
is that loop, done once:

* ``method="quadrature"`` -- Stroud's degree-3 rule (Stroud, 1957), the one
  Arndt (1996) and DeVuyst and Preckel (1997) brought to CGE models. For ``n``
  independent parameters with symmetric distributions it solves the model at
  ``2n`` points and returns a mean that is EXACT whenever the result is a
  polynomial of degree three or less in the parameters, and a standard
  deviation that is exact when the result is linear in them. Twenty
  elasticities cost forty solves.
* ``method="monte_carlo"`` -- independent draws with a fixed seed, for
  distributions that are not symmetric, for quantiles, or to check the
  quadrature.

Neither says where the variance comes from when parameters interact.
:func:`sobol` does: first-order and total Sobol indices on Saltelli's design,
at ``n (k + 2)`` solves for ``k`` parameters.

A run is any function from ``{parameter: value}`` to a number, a mapping of
numbers or a Series. :func:`spec_sweep` builds one from ``ModelSpec`` fields,
recalibrating at every point, because an elasticity moves the calibrated shares
and a result measured against an uncalibrated base would mix the two.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

__all__ = ["Uniform", "Normal", "Triangular", "SensitivityResult", "sensitivity",
           "stroud_points", "spec_sweep", "SobolResult", "sobol"]


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float
    symmetric = True

    def __post_init__(self):
        if not self.high > self.low:
            raise ValueError(f"Uniform needs high > low, got {self.low}, {self.high}")

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def sd(self) -> float:
        return (self.high - self.low) / np.sqrt(12.0)

    def draw(self, rng, n):
        return rng.uniform(self.low, self.high, n)

    def ppf(self, u):
        """The quantile at probability ``u``: how a point of the unit cube becomes a value."""
        return self.low + np.asarray(u, dtype=float) * (self.high - self.low)


@dataclass(frozen=True)
class Normal:
    mean: float
    sd: float
    symmetric = True

    def __post_init__(self):
        if not self.sd > 0:
            raise ValueError(f"Normal needs sd > 0, got {self.sd}")

    def draw(self, rng, n):
        return rng.normal(self.mean, self.sd, n)

    def ppf(self, u):
        from scipy.stats import norm
        return self.mean + self.sd * norm.ppf(np.asarray(u, dtype=float))


@dataclass(frozen=True)
class Triangular:
    low: float
    mode: float
    high: float

    def __post_init__(self):
        if not self.low <= self.mode <= self.high or not self.high > self.low:
            raise ValueError("Triangular needs low <= mode <= high and high > low")

    @property
    def symmetric(self) -> bool:
        return abs((self.mode - self.low) - (self.high - self.mode)) <= 1e-12 * (self.high - self.low)

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    @property
    def sd(self) -> float:
        a, c, b = self.low, self.mode, self.high
        return float(np.sqrt((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0))

    def draw(self, rng, n):
        return rng.triangular(self.low, self.mode, self.high, n)

    def ppf(self, u):
        a, c, b = self.low, self.mode, self.high
        u = np.asarray(u, dtype=float)
        cut = (c - a) / (b - a)
        with np.errstate(invalid="ignore"):
            left = a + np.sqrt(u * (b - a) * (c - a))
            right = b - np.sqrt((1.0 - u) * (b - a) * (b - c))
        return np.where(u < cut, left, right)


def stroud_points(n: int) -> np.ndarray:
    """Stroud's degree-3 points for ``n`` standardised variables, ``(2n, n)``.

    Each column has mean zero and variance one over the points, and every
    product of two different columns, and every odd moment, averages to zero.
    So with equal weights the rule integrates exactly any polynomial of degree
    three in independent variables with symmetric distributions.
    """
    if n < 1:
        raise ValueError("at least one parameter")
    pts = np.empty((2 * n, n))
    for k in range(1, 2 * n + 1):
        for r in range(1, n // 2 + 1):
            angle = (2 * r - 1) * k * np.pi / n
            pts[k - 1, 2 * r - 2] = np.sqrt(2.0) * np.cos(angle)
            pts[k - 1, 2 * r - 1] = np.sqrt(2.0) * np.sin(angle)
        if n % 2 == 1:
            pts[k - 1, n - 1] = (-1.0) ** k
    return pts


@dataclass
class SensitivityResult:
    """The points solved, their weights and outputs, and the summary over them."""

    method: str
    #: One row per point: parameter values, ``weight``, then the outputs.
    points: pd.DataFrame
    parameters: list
    outputs: list

    def summary(self, quantiles=(0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Mean and standard deviation of each output; quantiles under Monte Carlo.

        Quantiles from 2n quadrature points would be a fiction, so they are only
        reported for Monte Carlo draws.
        """
        y = self.points[self.outputs]
        w = self.points["weight"].to_numpy()
        mean = (y.mul(w, axis=0)).sum()
        var = ((y - mean) ** 2).mul(w, axis=0).sum()
        out = pd.DataFrame({"mean": mean, "sd": np.sqrt(var.clip(lower=0.0))})
        if self.method == "monte_carlo":
            for q in quantiles:
                out[f"q{q:g}"] = y.quantile(q)
            out["min"], out["max"] = y.min(), y.max()
        return out

    def __repr__(self) -> str:
        return (f"SensitivityResult({self.method}, {len(self.points)} points, "
                f"{len(self.parameters)} parameters, {len(self.outputs)} outputs)")


def _as_dict(value) -> dict:
    """Outputs as a plain dict; a pandas object per point is too slow for many draws."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return {"result": float(value)}
    if isinstance(value, Mapping) and not isinstance(value, pd.Series):
        return {k: float(v) for k, v in value.items()}
    return _as_series(value).to_dict()


def _as_series(value) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    if isinstance(value, Mapping):
        return pd.Series({k: float(v) for k, v in value.items()})
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return pd.Series({"result": float(arr[0])})
    return pd.Series(arr, index=[f"result_{i}" for i in range(arr.size)])


def sensitivity(run: Callable[[dict], object], parameters: Mapping[str, object], *,
                method: str = "quadrature", draws: int = 1000,
                seed: int | None = 20260724) -> SensitivityResult:
    """Solve ``run`` over the parameters' distributions and collect the results.

    ``parameters`` maps a name to :class:`Uniform`, :class:`Normal` or
    :class:`Triangular`. The parameters are independent. Quadrature needs every
    distribution symmetric, and refuses one that is not rather than return a
    mean with an unstated bias; Monte Carlo takes any of them.
    """
    if not parameters:
        raise ValueError("no parameters to vary")
    names = list(parameters)
    dists = [parameters[k] for k in names]
    if method == "quadrature":
        skewed = [k for k, d in zip(names, dists) if not d.symmetric]
        if skewed:
            raise ValueError(
                f"Stroud's rule assumes symmetric distributions; {skewed} are not. "
                f"Use method='monte_carlo' for them.")
        z = stroud_points(len(names))
        values = np.array([d.mean for d in dists])[None, :] + z * np.array(
            [d.sd for d in dists])[None, :]
        weights = np.full(len(z), 1.0 / len(z))
    elif method == "monte_carlo":
        if draws < 2:
            raise ValueError("at least two draws")
        rng = np.random.default_rng(seed)
        values = np.column_stack([d.draw(rng, draws) for d in dists])
        weights = np.full(draws, 1.0 / draws)
    else:
        raise ValueError(f"method must be 'quadrature' or 'monte_carlo', got {method!r}")

    rows = []
    reserved = set(names) | {"weight"}
    for v, w in zip(values, weights):
        point = dict(zip(names, map(float, v)))
        out = _as_dict(run(dict(point)))
        clash = reserved.intersection(out)
        if clash:
            raise ValueError(f"output names {sorted(clash)} collide with parameter names")
        rows.append({**point, "weight": float(w), **out})
    frame = pd.DataFrame(rows)
    outputs = [c for c in frame.columns if c not in names and c != "weight"]
    return SensitivityResult(method=method, points=frame, parameters=names, outputs=outputs)


def spec_sweep(sam, spec, *, measure: Callable, closure=None, scenario=None,
               calibrate_kwargs: Mapping | None = None) -> Callable[[dict], object]:
    """A run that sets ``ModelSpec`` fields, recalibrates, and solves base and scenario.

    A parameter named after a field (``"sigma_va"``) sets it whole; one named
    ``"field[account]"`` (``"sigma_armington[c_ind]"``) sets that account's
    entry and keeps the others. ``measure(base, alternative)`` turns the two
    solved periods into the outputs, for instance the percent change of real
    GDP. The base is solved again at every point because the calibration moved.
    """
    from .calibrate import calibrate
    from .solve import solve_period

    fields = {f.name for f in dataclasses.fields(spec)}
    kwargs = dict(calibrate_kwargs or {})

    def run(values: dict):
        changes: dict = {}
        for name, value in values.items():
            field, _, account = name.partition("[")
            if field not in fields:
                raise KeyError(f"{field!r} is not a ModelSpec field")
            if account:
                account = account.rstrip("]")
                current = changes.get(field, getattr(spec, field))
                if isinstance(current, Mapping):
                    mapping = dict(current)
                else:
                    mapping = {"__default__": float(current)}
                mapping[account] = float(value)
                changes[field] = mapping
            else:
                changes[field] = float(value)
        econ = calibrate(sam, dataclasses.replace(spec, **changes), **kwargs)
        base = solve_period(econ, closure)
        alt = solve_period(econ, closure, scenario, warm_start=base)
        return measure(base, alt)

    return run


# ---------------------------------------------------------------------------
# Sobol indices
# ---------------------------------------------------------------------------

@dataclass
class SobolResult:
    """The points of Saltelli's design, their outputs, and the Sobol indices."""

    #: One row per solve: ``matrix`` (``"A"``, ``"B"`` or ``"AB_<parameter>"``),
    #: ``row`` (the base point it came from), the parameter values, ``weight``
    #: and the outputs. The ``A`` and ``B`` rows are independent draws, each
    #: weighted ``1 / (2n)``; the ``AB`` rows carry weight zero. As in
    #: :class:`SensitivityResult`, the weighted rows are a sample of the result's
    #: distribution.
    points: pd.DataFrame
    parameters: list
    outputs: list
    n: int
    bootstrap: int
    conf_level: float
    seed: int | None

    def _arrays(self, output):
        p = self.points
        y = lambda m: p.loc[p["matrix"] == m, output].to_numpy(dtype=float)  # noqa: E731
        A, B = y("A"), y("B")
        AB = np.column_stack([y(f"AB_{k}") for k in self.parameters])
        return A, B, AB

    def indices(self) -> pd.DataFrame:
        """``S1``, ``ST`` and their half-widths at ``conf_level``, by output and parameter.

        First order by Saltelli et al. (2010), total by Jansen (1999), each over
        the variance of the ``A`` and ``B`` outputs together, on outputs centred
        on the mean of every solve: the estimators of SALib, which give the same
        numbers on the same design. The half-widths come from a bootstrap over
        the base points.
        """
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + self.conf_level / 2.0))
        rng = np.random.default_rng(self.seed)
        draws = rng.integers(0, self.n, size=(self.bootstrap, self.n))
        frames = []
        for out in self.outputs:
            A, B, AB = self._arrays(out)
            # Centred on the mean of every solve, as SALib does: the estimators
            # are the same in exact arithmetic, and the first-order one loses
            # less to cancellation when the result sits far from zero.
            centre = np.mean(np.concatenate([A, B, AB.ravel()]))
            A, B, AB = A - centre, B - centre, AB - centre
            s1, st = _sobol_indices(A, B, AB)
            if self.bootstrap:
                boot = [_sobol_indices(A[r], B[r], AB[r]) for r in draws]
                s1_conf = z * np.std([b[0] for b in boot], axis=0, ddof=1)
                st_conf = z * np.std([b[1] for b in boot], axis=0, ddof=1)
            else:
                s1_conf = st_conf = np.full(len(self.parameters), np.nan)
            frames.append(pd.DataFrame({"output": out, "parameter": self.parameters,
                                        "S1": s1, "S1_conf": s1_conf,
                                        "ST": st, "ST_conf": st_conf}))
        return pd.concat(frames, ignore_index=True).set_index(["output", "parameter"])

    def summary(self, quantiles=(0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Mean, standard deviation and quantiles of each output over the ``A`` and ``B`` draws."""
        y = self.points.loc[self.points["matrix"].isin(["A", "B"]), self.outputs]
        out = pd.DataFrame({"mean": y.mean(), "sd": y.std(ddof=0)})
        for q in quantiles:
            out[f"q{q:g}"] = y.quantile(q)
        return out

    def __repr__(self) -> str:
        return (f"SobolResult({len(self.points)} solves, n={self.n}, "
                f"{len(self.parameters)} parameters, {len(self.outputs)} outputs)")


def _sobol_indices(A, B, AB):
    """First-order (Saltelli 2010) and total (Jansen 1999) indices of each column of ``AB``."""
    var = np.var(np.concatenate([A, B]))
    if not var > 0:
        k = AB.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan)
    s1 = np.mean(B[:, None] * (AB - A[:, None]), axis=0) / var
    st = 0.5 * np.mean((A[:, None] - AB) ** 2, axis=0) / var
    return s1, st


def sobol(run: Callable[[dict], object], parameters: Mapping[str, object], *,
          n: int = 1024, seed: int | None = 20260724, bootstrap: int = 200,
          conf_level: float = 0.95) -> SobolResult:
    """First-order and total Sobol indices of ``run`` over independent parameters.

    Saltelli's design: two matrices ``A`` and ``B`` of ``n`` points each, from a
    scrambled Sobol sequence of dimension ``2k`` (``scipy.stats.qmc``), carried
    to each parameter's distribution by its quantile function; and, for each
    parameter ``i``, the matrix ``AB_i``, ``A`` with column ``i`` taken from
    ``B``. That is ``n (k + 2)`` solves. ``n`` must be a power of two, which is
    what keeps a Sobol sequence balanced.

    The first-order index ``S1`` is the share of the variance that parameter
    explains alone; the total index ``ST`` adds every interaction it takes part
    in, so ``ST - S1`` is how much of it comes from interactions. ``run`` and
    ``parameters`` are those of :func:`sensitivity`; :func:`spec_sweep` builds
    the run.
    """
    from scipy.stats import qmc

    if not parameters:
        raise ValueError("no parameters to vary")
    if n < 2 or n & (n - 1):
        raise ValueError(f"n must be a power of two, at least 2, got {n}: a Sobol sequence "
                         f"is balanced only in powers of two")
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must lie strictly between 0 and 1, got {conf_level}")
    if bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero (no intervals) or at least two resamples")
    names = list(parameters)
    dists = [parameters[k] for k in names]
    missing = [k for k, d in zip(names, dists) if not hasattr(d, "ppf")]
    if missing:
        raise TypeError(f"{missing} have no quantile function (ppf)")
    k = len(names)
    base = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(n)
    to_values = lambda u: np.column_stack(  # noqa: E731
        [d.ppf(u[:, j]) for j, d in enumerate(dists)])
    A, B = to_values(base[:, :k]), to_values(base[:, k:])
    blocks = [("A", A), ("B", B)]
    for i, name in enumerate(names):
        AB = A.copy()
        AB[:, i] = B[:, i]
        blocks.append((f"AB_{name}", AB))

    rows = []
    reserved = set(names) | {"weight", "matrix", "row"}
    for label, values in blocks:
        weight = 1.0 / (2 * n) if label in ("A", "B") else 0.0
        for j, v in enumerate(values):
            point = dict(zip(names, map(float, v)))
            out = _as_dict(run(dict(point)))
            clash = reserved.intersection(out)
            if clash:
                raise ValueError(f"output names {sorted(clash)} collide with parameter names")
            rows.append({"matrix": label, "row": j, **point, "weight": weight, **out})
    frame = pd.DataFrame(rows)
    outputs = [c for c in frame.columns if c not in reserved]
    return SobolResult(points=frame, parameters=names, outputs=outputs, n=n,
                       bootstrap=int(bootstrap), conf_level=float(conf_level), seed=seed)
