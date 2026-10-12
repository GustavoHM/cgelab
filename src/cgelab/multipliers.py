"""SAM multipliers and input-output analysis, read straight from the base year.

No equilibrium is solved here. A multiplier model fixes every price and lets
quantities answer to an injection into the exogenous accounts, so it is the
linear, demand-driven reading of the same SAM a CGE model calibrates on. It is
worth having next to the CGE for two reasons: it is what a reader of an applied
study will compare the CGE result against, and when the two disagree the
difference is the work done by prices and supply constraints.

* :func:`accounting_multipliers` -- Pyatt and Round (1979). The endogenous
  accounts spend in base-year average proportions; ``M = (I - A)^-1``.
* :func:`fixed_price_multipliers` -- the same, with household consumption at
  MARGINAL propensities built from income elasticities.
* :func:`leontief_multipliers` -- only activities and commodities endogenous, so
  the activity block of ``M`` is the domestic Leontief inverse.
* :func:`linkages` -- backward linkages from ``M`` (Rasmussen) and forward
  linkages from the Ghosh inverse, each with its normalised index.
* :func:`structural_paths` -- Defourny and Thorbecke (1984): how an injection
  into one account reaches another, path by path.

Convention: ``A[i, j]`` is what account ``j`` pays account ``i`` per unit of
``j``'s total outlay, so ``M[i, j]`` is the effect on ``i`` of one unit injected
into ``j``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sam import SAM

__all__ = ["Multipliers", "accounting_multipliers", "fixed_price_multipliers",
           "leontief_multipliers", "linkages", "structural_paths"]


@dataclass
class Multipliers:
    """A multiplier matrix and what it was built from."""

    kind: str
    accounts: list
    #: Base-year flows among the endogenous accounts, ``T[i, j]`` paid by ``j`` to ``i``.
    flows: pd.DataFrame
    #: Column totals of the endogenous accounts.
    totals: pd.Series
    #: Propensities the inverse was taken of: average, or marginal where stated.
    A: pd.DataFrame
    M: pd.DataFrame

    @property
    def leakage(self) -> pd.Series:
        """Share of each account's outlay that goes to the exogenous accounts."""
        return (1.0 - self.A.sum(axis=0)).rename("leakage")

    def impact(self, injection: Mapping[str, float]) -> pd.Series:
        """Change in every endogenous account's total for an exogenous injection."""
        x = pd.Series(0.0, index=self.accounts)
        for name, value in injection.items():
            if name not in x.index:
                raise KeyError(f"{name!r} is not an endogenous account; the injection "
                               f"enters through the exogenous accounts' payments to one")
            x[name] += float(value)
        return (self.M @ x).rename("impact")

    def __repr__(self) -> str:
        return f"Multipliers({self.kind}, {len(self.accounts)} endogenous accounts)"


def _endogenous(sam: SAM, endogenous, exogenous, spec) -> list:
    if endogenous is not None and exogenous is not None:
        raise ValueError("pass endogenous or exogenous, not both")
    acc = sam.accounts
    if endogenous is not None:
        names = list(endogenous)
        missing = [a for a in names if a not in acc]
        if missing:
            raise KeyError(f"accounts not in the SAM: {missing}")
        return names
    if exogenous is None:
        if spec is None:
            raise ValueError(
                "say which accounts are exogenous: pass exogenous=, endogenous=, or a "
                "ModelSpec, whose government, investment and rest-of-world accounts "
                "(and tariff and indirect-tax accounts) are then exogenous")
        exogenous = [spec.government, spec.investment, spec.rest_of_world]
        exogenous += [a for a in (spec.tariff_account, spec.indirect_tax_account) if a]
    exogenous = list(exogenous)
    missing = [a for a in exogenous if a not in acc]
    if missing:
        raise KeyError(f"accounts not in the SAM: {missing}")
    return [a for a in acc if a not in set(exogenous)]


def _build(sam: SAM, names: list, kind: str, A=None) -> Multipliers:
    T = sam.block(names, names)
    y = sam.outlays().reindex(names).to_numpy(dtype=float)
    if A is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(y[None, :] != 0, T / np.where(y == 0, 1.0, y)[None, :], 0.0)
    eye = np.eye(len(names))
    try:
        M = np.linalg.solve(eye - A, eye)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "I - A is singular: some set of endogenous accounts spends everything it "
            "receives inside the set, so nothing leaks to an exogenous account and an "
            "injection has no finite effect. Make one of them exogenous.") from exc
    frame = lambda m: pd.DataFrame(m, index=names, columns=names)  # noqa: E731
    return Multipliers(kind=kind, accounts=names, flows=frame(T),
                       totals=pd.Series(y, index=names, name="total"), A=frame(A), M=frame(M))


def accounting_multipliers(sam: SAM, *, exogenous: Sequence[str] | None = None,
                           endogenous: Sequence[str] | None = None, spec=None) -> Multipliers:
    """Pyatt-Round accounting multipliers: ``M = (I - A)^-1`` with average propensities.

    The usual split makes government, investment and the rest of the world
    exogenous. That is what ``spec`` gives; ``exogenous`` or ``endogenous``
    states any other.
    """
    return _build(sam, _endogenous(sam, endogenous, exogenous, spec), "accounting")


def fixed_price_multipliers(sam: SAM, *, households: Sequence[str],
                            income_elasticities: Mapping[str, float],
                            exogenous: Sequence[str] | None = None,
                            endogenous: Sequence[str] | None = None, spec=None,
                            engel: str = "check") -> Multipliers:
    """Pyatt-Round fixed-price multipliers.

    A household spends an extra unit of income on commodity ``c`` in proportion
    ``eta_c * a_ch``: the income elasticity times the average propensity. Every
    other column keeps its average propensities. Commodities absent from
    ``income_elasticities`` have elasticity one, so all-ones gives back the
    accounting multipliers.

    The marginal propensities of a household must add up to its average
    propensity to consume (Engel aggregation), or an extra unit of income would
    create or destroy spending. ``engel="check"`` refuses elasticities that
    violate it; ``"rescale"`` scales them, household by household, until they do.
    """
    if engel not in ("check", "rescale"):
        raise ValueError(f"engel must be 'check' or 'rescale', got {engel!r}")
    names = _endogenous(sam, endogenous, exogenous, spec)
    base = _build(sam, names, "accounting")
    A = base.A.to_numpy().copy()
    idx = {a: i for i, a in enumerate(names)}
    unknown = [c for c in income_elasticities if c not in idx]
    if unknown:
        raise KeyError(f"income elasticities for accounts that are not endogenous: {unknown}")
    rows = [idx[c] for c in income_elasticities]
    eta = np.array([float(income_elasticities[c]) for c in income_elasticities])
    for h in households:
        if h not in idx:
            raise KeyError(f"household {h!r} is not an endogenous account")
        j = idx[h]
        average = A[rows, j]
        marginal = eta * average
        spent, extra = float(average.sum()), float(marginal.sum())
        if abs(extra - spent) > 1e-9 * max(abs(spent), 1e-12):
            if engel == "check":
                raise ValueError(
                    f"household {h!r}: marginal propensities sum to {extra:.6g} against an "
                    f"average propensity to consume of {spent:.6g}. The elasticities break "
                    f"Engel aggregation for this budget; fix them, or pass engel='rescale'.")
            marginal = marginal * (spent / extra)
        A[rows, j] = marginal
    out = _build(sam, names, "fixed_price", A=A)
    return out


def leontief_multipliers(sam: SAM, spec) -> Multipliers:
    """Only activities and commodities endogenous.

    The activity block of ``M`` is the domestic Leontief inverse: commodity
    columns leak imports and taxes, activity columns leak value added.
    """
    layout = spec.prepare(sam).classify(sam)
    return _build(sam, list(layout.activities) + list(layout.commodities), "leontief")


def linkages(mult: Multipliers, accounts: Sequence[str] | None = None) -> pd.DataFrame:
    """Backward and forward linkages of each account, raw and normalised.

    Backward linkage of ``j``: the column sum of ``M`` over ``accounts``, what
    one unit of final demand for ``j`` pulls from them. Forward linkage of
    ``i``: the row sum of the Ghosh inverse ``(I - B)^-1``, ``B[i, j] =
    T[i, j] / total_i``, what one unit of primary input into ``i`` pushes into
    them. The Ghosh row sum, not the Leontief one, is the forward measure Miller
    and Blair recommend: the Leontief row sum answers a uniform rise in every
    final demand, which is a different question.

    The indices divide each linkage by the mean over ``accounts`` (Rasmussen's
    power and sensitivity of dispersion). Above one is above average.
    ``accounts`` defaults to every endogenous account; pass the activities to
    read the classic key-sector table.
    """
    names = mult.accounts if accounts is None else list(accounts)
    unknown = [a for a in names if a not in mult.accounts]
    if unknown:
        raise KeyError(f"accounts not endogenous in these multipliers: {unknown}")
    T = mult.flows.to_numpy()
    y = mult.totals.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(y[:, None] != 0, T / np.where(y == 0, 1.0, y)[:, None], 0.0)
    eye = np.eye(len(mult.accounts))
    G = pd.DataFrame(np.linalg.solve(eye - B, eye), index=mult.accounts, columns=mult.accounts)
    backward = mult.M.loc[names, names].sum(axis=0)
    forward = G.loc[names, names].sum(axis=1)
    return pd.DataFrame({
        "backward": backward, "backward_index": backward / backward.mean(),
        "forward": forward, "forward_index": forward / forward.mean(),
    })


def structural_paths(mult: Multipliers, origin: str, destination: str | None = None, *,
                     max_length: int = 6, threshold: float = 1e-4) -> pd.DataFrame:
    """Elementary paths from ``origin`` to ``destination``, with their influences.

    Defourny and Thorbecke (1984). Along a path ``origin -> ... -> destination``
    the DIRECT influence is the product of the propensities on its arcs. The
    TOTAL influence multiplies it by the path multiplier ``det(I - A_p) /
    det(I - A)``, where ``A_p`` drops the path's accounts: the feedback that
    returns to the path through circuits off it. The GLOBAL influence is
    ``M[destination, origin]``, and it equals the sum of the total influences of
    every elementary path (Lantner, 1974); ``share`` is each path's part of it.

    Paths are cut when their direct influence falls below ``threshold`` or
    their length passes ``max_length``, so on a real SAM the shares do not sum
    to one. Without ``destination``, every other endogenous account is one.
    """
    names = mult.accounts
    idx = {a: i for i, a in enumerate(names)}
    if origin not in idx:
        raise KeyError(f"{origin!r} is not an endogenous account")
    targets = [d for d in names if d != origin] if destination is None else [destination]
    for d in targets:
        if d not in idx:
            raise KeyError(f"{d!r} is not an endogenous account")
        if d == origin:
            raise ValueError("a structural path runs between two different accounts")
    A = mult.A.to_numpy()
    n = len(names)
    eye = np.eye(n)
    sign0, logdet0 = np.linalg.slogdet(eye - A)
    target_set = {idx[d] for d in targets}
    arcs = [np.flatnonzero(A[:, j]) for j in range(n)]

    def path_multiplier(nodes):
        keep = np.setdiff1d(np.arange(n), nodes)
        if keep.size == 0:
            sign, logdet = 1.0, 0.0
        else:
            sign, logdet = np.linalg.slogdet(eye[np.ix_(keep, keep)] - A[np.ix_(keep, keep)])
        return float(sign * sign0 * np.exp(logdet - logdet0))

    rows = []
    stack = [([idx[origin]], 1.0)]
    while stack:
        nodes, direct = stack.pop()
        here = nodes[-1]
        if len(nodes) - 1 >= max_length:
            continue
        for nxt in arcs[here]:
            if nxt in nodes:
                continue
            d = direct * A[nxt, here]
            if abs(d) < threshold:
                continue
            path = nodes + [int(nxt)]
            if nxt in target_set:
                pm = path_multiplier(path)
                dest = names[nxt]
                rows.append({"origin": origin, "destination": dest,
                             "path": tuple(names[k] for k in path), "length": len(path) - 1,
                             "direct": d, "path_multiplier": pm, "total": d * pm,
                             "global": float(mult.M.iat[nxt, idx[origin]])})
            stack.append((path, d))
    cols = ["origin", "destination", "path", "length", "direct", "path_multiplier",
            "total", "global", "share"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["share"] = out["total"] / out["global"]
    return (out.sort_values(["destination", "total"], ascending=[True, False],
                            key=lambda s: s.abs() if s.name == "total" else s)
            .reset_index(drop=True)[cols])
