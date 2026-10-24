"""Uncertainty in the SAM itself, propagated through calibration and solution.

A SAM is an estimate, not a set of exact constants. :func:`sam_draws` perturbs
selected blocks and rebalances the matrix; :func:`sam_sweep` recalibrates and
solves every draw. The points have the same shape as sensitivity draws, so
coverage and probability-integral-transform calculations can consume them
without another adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .sam import SAM, sam_distance
from .sensitivity import SensitivityResult, _as_dict, spec_sweep

__all__ = ["SAMDraws", "sam_draws", "sam_sweep"]


@dataclass
class SAMDraws:
    """Balanced SAM draws and their sampling weights.

    ``points`` has one row per draw: ``draw``, ``weight`` and ``sam_distance``.
    The distance is ``D_KL(original || draw)``. It is NaN when negative flows
    make Kullback--Leibler divergence undefined.
    """

    sams: list[SAM] = field(repr=False)
    points: pd.DataFrame
    method: str
    blocks: dict = field(repr=False)
    seed: int | None

    def __len__(self):
        return len(self.sams)

    def __repr__(self):
        return f"SAMDraws({self.method}, {len(self.sams)} draws)"


def _blocks(blocks, accounts):
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("blocks must be a nonempty mapping of labels to (rows, columns)")
    known = set(accounts)
    out = {}
    occupied = set()
    for label, specification in blocks.items():
        if not isinstance(label, str) or not label:
            raise ValueError("every SAM block needs a nonempty string label")
        try:
            rows, columns = specification
            rows, columns = list(rows), list(columns)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"block {label!r} must be a (rows, columns) pair") from exc
        if not rows or not columns:
            raise ValueError(f"block {label!r} has no rows or no columns")
        if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
            raise ValueError(f"block {label!r} repeats an account")
        missing = sorted((set(rows) | set(columns)) - known)
        if missing:
            raise KeyError(f"block {label!r} names accounts not in the SAM: {missing}")
        cells = {(r, c) for r in rows for c in columns}
        overlap = sorted(occupied & cells)
        if overlap:
            raise ValueError(f"block {label!r} overlaps another block at {overlap[:3]}")
        occupied |= cells
        out[label] = (rows, columns)
    return out


def sam_draws(sam: SAM, *, blocks, method: str = "dirichlet",
              concentration: float = 100.0, scale: float = 0.05,
              draws: int = 1000, seed: int | None = 20260724,
              balance_kwargs: Mapping | None = None) -> SAMDraws:
    """Draw balanced SAMs by perturbing the given blocks.

    ``blocks`` maps a label to ``(rows, columns)``. Blocks may not overlap. In
    ``"dirichlet"`` mode, the positive selected cells of each column follow a
    Dirichlet distribution centred on their original shares, with precision
    ``concentration``. Structural zeros stay zero, and negative selected cells
    are refused. In ``"noise"`` mode, every nonzero selected cell is multiplied
    by independent lognormal noise with mean one and log standard deviation
    ``scale``.

    The blocks delimit the direct shock, not the cells that balancing may move:
    every draw is rebalanced by :meth:`SAM.balance`, using ``balance_kwargs``.
    The input SAM is neither mutated nor used if it is initially unbalanced.
    """
    if method not in ("dirichlet", "noise"):
        raise ValueError(f"method must be 'dirichlet' or 'noise', got {method!r}")
    draws = int(draws)
    if draws < 1:
        raise ValueError("draws must be at least one")
    concentration = float(concentration)
    scale = float(scale)
    if method == "dirichlet" and not (np.isfinite(concentration) and concentration > 0):
        raise ValueError("concentration must be a positive finite number")
    if method == "noise" and not (np.isfinite(scale) and scale >= 0):
        raise ValueError("scale must be a nonnegative finite number")
    block_spec = _blocks(blocks, sam.accounts)
    sam.check_balance()
    kwargs = dict(balance_kwargs or {})
    rng = np.random.default_rng(seed)
    distance_defined = bool((sam.frame.value >= 0).all() and np.isfinite(sam.frame.value).all())
    sams, rows = [], []

    for i in range(draws):
        frame = sam.frame.copy(deep=True).reset_index(drop=True)
        for label, (block_rows, block_columns) in block_spec.items():
            in_block = frame.row.isin(block_rows) & frame.col.isin(block_columns)
            if method == "dirichlet":
                if (frame.loc[in_block, "value"] < 0).any():
                    raise ValueError(f"block {label!r} has negative entries; Dirichlet shares "
                                     "require nonnegative flows, or use method='noise'")
                for column in block_columns:
                    selected = frame.index[in_block & (frame.col == column) & (frame.value > 0)]
                    values = frame.loc[selected, "value"].to_numpy(dtype=float)
                    total = values.sum()
                    if total > 0:
                        shares = rng.dirichlet(concentration * values / total)
                        frame.loc[selected, "value"] = total * shares
            else:
                selected = frame.index[in_block & (frame.value != 0)]
                noise = rng.lognormal(mean=-0.5 * scale ** 2, sigma=scale, size=len(selected))
                frame.loc[selected, "value"] *= noise
        candidate = SAM.from_long(frame, name=f"{sam.name}-{method}-{i}")
        balanced = candidate.balance(name=candidate.name, **kwargs)
        balanced.check_balance(**{k: v for k, v in kwargs.items()
                                  if k in ("rel_tol", "abs_tol")})
        sams.append(balanced)
        rows.append({"draw": i, "weight": 1.0 / draws,
                     "sam_distance": sam_distance(sam, balanced) if distance_defined else np.nan})
    return SAMDraws(sams=sams, points=pd.DataFrame(rows), method=method,
                    blocks=block_spec, seed=seed)


def sam_sweep(draws: SAMDraws, spec, *, measure: Callable, closure=None, scenario=None,
              parameters: Mapping[str, object] | None = None,
              calibrate_kwargs: Mapping | None = None) -> SensitivityResult:
    """Recalibrate and solve every SAM draw, optionally with elasticity draws.

    ``measure(reference, alternative)`` has the same meaning as in
    :func:`cgelab.spec_sweep`. ``parameters`` maps ModelSpec fields to the same
    distributions accepted by :func:`cgelab.sensitivity`; one independent value
    per field is drawn on each SAM row. The returned ``SensitivityResult`` uses
    Monte Carlo summaries. Its ``points`` retain ``draw``, ``weight`` and
    ``sam_distance`` alongside the parameter values and outputs.
    """
    if not isinstance(draws, SAMDraws):
        raise TypeError("draws must be the SAMDraws returned by sam_draws")
    parameters = dict(parameters or {})
    missing = [name for name, distribution in parameters.items()
               if not hasattr(distribution, "draw")]
    if missing:
        raise TypeError(f"parameters {missing} have no draw method")
    rng = np.random.default_rng(draws.seed)
    records = []
    reserved = {"draw", "weight", "sam_distance"} | set(parameters)
    for i, drawn_sam in enumerate(draws.sams):
        values = {name: float(np.asarray(distribution.draw(rng, 1)).ravel()[0])
                  for name, distribution in parameters.items()}
        try:
            run = spec_sweep(drawn_sam, spec, measure=measure, closure=closure,
                             scenario=scenario, calibrate_kwargs=calibrate_kwargs)
            output = _as_dict(run(values))
        except Exception as exc:
            raise RuntimeError(f"SAM draw {i} failed: {exc}") from exc
        clash = reserved.intersection(output)
        if clash:
            raise ValueError(f"output names {sorted(clash)} collide with SAM draw fields")
        records.append({"draw": i, "weight": float(draws.points.iloc[i]["weight"]),
                        "sam_distance": float(draws.points.iloc[i]["sam_distance"]),
                        **values, **output})
    points = pd.DataFrame(records)
    points.attrs["sam_method"] = draws.method
    outputs = [column for column in points.columns if column not in reserved]
    return SensitivityResult(method="monte_carlo", points=points,
                             parameters=list(parameters), outputs=outputs)
