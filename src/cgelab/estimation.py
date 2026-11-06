"""Estimating a SAM by cross entropy, and regionalising national coefficients.

**Cross entropy.** A prior SAM that does not balance, or that has to meet information it was
not built from (account totals from the national accounts, a cell known from a survey, an
aggregate), is moved to the balanced SAM closest to it in the sense of information:
the minimum of Csiszár's I-divergence over the flows,

    sum_ij  T_ij log(T_ij / P_ij) - T_ij + P_ij,

subject to every account's receipts equalling its outlays and to the information given. This is
the cross-entropy estimation of Golan, Judge and Robinson (1994) and Robinson, Cattaneo and
El-Said (2001) written on flows. The solution keeps every zero of the prior and every sign:
``T_ij = P_ij exp(mu_j - mu_i - sum_k nu_k G_kij)``, one multiplier per account balance and one
per piece of information. When the total of every account is given, the problem is RAS's and so
is the solution (Bacharach, 1970); with no totals given, the balance alone scales each account's
row and column by reciprocal factors. The multipliers are found by Newton's method on the dual,
which is smooth and convex.

A negative entry, which no divergence admits, is moved to the transposed cell with its sign
changed: ``-x`` paid from ``j`` to ``i`` is ``x`` paid from ``i`` to ``j``, and every account's
balance is the same either way. It is moved back afterwards.

**Location quotients.** Regional input coefficients from national ones, ``r_ij = a_ij
min(1, LQ_ij)``, with the simple (SLQ), cross-industry (CILQ) and Flegg (FLQ, AFLQ) quotients
(Flegg, Webber and Elliott, 1995; Flegg and Webber, 2000; McCann and Dewhurst, 1998).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .sam import SAM, BalanceError

__all__ = ["CrossEntropy", "cross_entropy", "location_quotients", "regional_coefficients"]


@dataclass
class CrossEntropy:
    """A SAM estimated by :func:`cross_entropy`, with what the estimation says about it.

    ``divergence`` is the I-divergence from the prior; ``balance_multipliers`` one per account
    (normalised so the first is zero) and ``information_multipliers`` one per total, aggregate
    and cell target, in the order given: the shadow value of each piece of information, how
    much the divergence would fall without it. ``iterations`` are Newton steps.
    """

    sam: SAM
    divergence: float
    balance_multipliers: pd.Series
    information_multipliers: pd.Series
    iterations: int


def cross_entropy(prior: SAM, *, totals: Mapping[str, float] | None = None,
                  fixed: Iterable[tuple[str, str]] = (),
                  targets: Mapping[tuple[str, str], float] | None = None,
                  aggregates: Sequence[tuple[Mapping[tuple[str, str], float], float]] = (),
                  tol: float = 1e-12, max_iter: int = 200, name: str | None = None) -> CrossEntropy:
    """The balanced SAM closest to ``prior`` that meets the information given.

    Parameters
    ----------
    totals : mapping, optional
        Account -> its total (receipts, which balance makes equal to outlays).
    fixed : iterable of (row, col)
        Cells held at their prior value.
    targets : mapping, optional
        (row, col) -> a value the cell must take. The cell must be nonzero in the prior.
    aggregates : sequence of (weights, value)
        Linear information on the flows: ``sum weights[(row, col)] * T[row, col] = value``,
        a GDP or a block total.

    The prior's zeros stay zero. Information that the prior's structure cannot meet (a target
    on a cell the prior leaves empty, totals that contradict each other) has no solution, and
    the estimation raises rather than returning the nearest miss.
    """
    acc = list(prior.accounts)
    index = {a: k for k, a in enumerate(acc)}
    frame = prior.frame
    if not np.isfinite(frame.value.to_numpy(dtype=float)).all():
        prior.check_balance()                    # a missing flow cannot be estimated around
    rows = frame.row.map(index).to_numpy()
    cols = frame.col.map(index).to_numpy()
    vals = frame.value.to_numpy(dtype=float)
    keep = vals != 0.0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    # A negative flow is the opposite payment in the transposed cell.
    flip = vals < 0
    r = np.where(flip, cols, rows)
    c = np.where(flip, rows, cols)
    p = np.abs(vals)
    if np.any(r == c):
        diag = [acc[k] for k in r[r == c]]
        raise ValueError(f"cross_entropy: diagonal cells {diag} are on both sides of their "
                         f"account's balance; drop them before estimating")
    cell = {}
    for k, (i, j) in enumerate(zip(r, c)):
        if (i, j) in cell:
            raise ValueError(
                f"cross_entropy: ({acc[i]!r}, {acc[j]!r}) and its transposed cell are both in the "
                f"prior, one of them negative; net them into one cell first")
        cell[(i, j)] = k
    orig = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}

    def locate(rc, what):
        a, b = rc
        if a not in index or b not in index:
            raise KeyError(f"cross_entropy: {what} names ({a!r}, {b!r}), which is not in the SAM")
        key = (index[a], index[b])
        if key not in orig:
            raise ValueError(f"cross_entropy: {what} names ({a!r}, {b!r}), which the prior "
                             f"leaves empty; cross entropy keeps the prior's zeros")
        k = orig[key]
        return k, (-1.0 if flip[k] else 1.0)

    n, m = len(acc), p.size
    is_fixed = np.zeros(m, dtype=bool)
    for rc in fixed:
        k, _ = locate(rc, "fixed")
        is_fixed[k] = True
    # Information as rows of G (on the stored, sign-flipped flows) and g.
    G_rows, g, labels = [], [], []
    for a, value in (totals or {}).items():
        if a not in index:
            raise KeyError(f"cross_entropy: totals names {a!r}, which is not an account")
        i = index[a]
        G_rows.append((r == i).astype(float))     # receipts of account i
        g.append(float(value))
        labels.append(f"total:{a}")
    for rc, value in (targets or {}).items():
        k, s = locate(rc, "targets")
        row = np.zeros(m)
        row[k] = s
        G_rows.append(row)
        g.append(float(value))
        labels.append(f"cell:{rc[0]}<-{rc[1]}")
    for q, (weights, value) in enumerate(aggregates):
        row = np.zeros(m)
        for rc, wgt in weights.items():
            k, s = locate(rc, "aggregates")
            row[k] += s * float(wgt)
        G_rows.append(row)
        g.append(float(value))
        labels.append(f"aggregate:{q}")
    G = np.array(G_rows).reshape(len(G_rows), m)
    g = np.array(g)
    # Fixed cells are constants: they enter the balances and the information as known amounts.
    free = ~is_fixed
    fixed_val = np.where(is_fixed, p, 0.0)
    b_fix = np.bincount(r, fixed_val, n) - np.bincount(c, fixed_val, n)   # receipts - outlays
    g_free = g - G @ fixed_val
    Gf = G[:, free]
    rf, cf, pf = r[free], c[free], p[free]
    nk = len(g)
    # Unknowns: mu_1..mu_{n-1} (mu_0 = 0, the balances' redundancy) and nu.
    D = np.zeros((n, rf.size))                    # d(receipts - outlays)/dT, by account
    D[rf, np.arange(rf.size)] += 1.0
    D[cf, np.arange(rf.size)] -= 1.0
    A = np.vstack([D[1:], Gf])                    # constraint rows on the free flows
    rhs = np.concatenate([-b_fix[1:], g_free])

    def flows(x):
        mu = np.concatenate([[0.0], x[:n - 1]])
        nu = x[n - 1:]
        expo = mu[cf] - mu[rf] - (nu @ Gf if nk else 0.0)
        return pf * np.exp(np.clip(expo, -700, 700))

    def dual(x):
        # phi(x) = sum T(x) + x . rhs, convex; its gradient is rhs - A T.
        return float(flows(x).sum() + x @ rhs)

    x = np.zeros(n - 1 + nk)
    scale = max(float(np.abs(p).max()), 1.0)
    it = 0
    for it in range(1, max_iter + 1):
        t = flows(x)
        grad = rhs - A @ t
        if np.max(np.abs(grad)) <= tol * scale:
            break
        H = (A * t[None, :]) @ A.T
        # The dual's gradient is rhs - A T and its Hessian A diag(T) A'.
        try:
            step = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(H, grad, rcond=None)[0]
        f0, alpha, slope = dual(x), 1.0, float(grad @ step)
        gnorm = float(np.max(np.abs(grad)))
        while alpha > 1e-12:
            trial = x + alpha * step
            f1 = dual(trial)
            if f1 <= f0 + 1e-4 * alpha * slope:
                break
            # Near the solution the fall in the dual is below its rounding error: accept a
            # step that brings the balances and the information closer instead.
            if abs(f1 - f0) <= 1e-12 * max(abs(f0), 1.0) and \
                    np.max(np.abs(rhs - A @ flows(trial))) < gnorm:
                break
            alpha *= 0.5
        if alpha <= 1e-12:
            break
        x = x + alpha * step
    t = flows(x)
    gap = rhs - A @ t
    if not np.max(np.abs(gap)) <= max(tol * scale * 10, 1e-9 * scale):
        raise BalanceError(
            f"cross_entropy did not meet the balances and the information (worst gap "
            f"{np.max(np.abs(gap)):.3g} after {it} Newton steps). The information may be "
            f"inconsistent with the prior's zeros or with itself.")
    est = p.copy()
    est[free] = t
    out = frame[keep].copy().reset_index(drop=True)
    out["value"] = np.where(flip, -est, est)
    sam = SAM(frame=out, name=name or f"{prior.name}-ce")
    with np.errstate(divide="ignore", invalid="ignore"):
        div = float(np.sum(est * np.log(est / p) - est + p))
    mu = np.concatenate([[0.0], x[:n - 1]])
    return CrossEntropy(sam=sam, divergence=div,
                        balance_multipliers=pd.Series(mu, index=acc, name="balance"),
                        information_multipliers=pd.Series(x[n - 1:], index=labels,
                                                          name="information"),
                        iterations=it)


def location_quotients(regional: Mapping[str, float] | pd.Series,
                       national: Mapping[str, float] | pd.Series, *, method: str = "flq",
                       delta: float = 0.3) -> pd.DataFrame:
    """Location quotients by supplying (rows) and purchasing (columns) sector.

    ``regional`` and ``national`` are a size measure by sector, usually employment or output.
    ``method``:

    * ``"slq"``, simple: ``SLQ_i = (E_i^r / E^r) / (E_i^n / E^n)``, the same in every column;
    * ``"cilq"``, cross-industry: ``SLQ_i / SLQ_j``, with ``SLQ_i`` on the diagonal;
    * ``"flq"``, Flegg's: the CILQ (and the SLQ on the diagonal) times
      ``lambda = log2(1 + E^r / E^n) ** delta``, which shrinks the quotients of a small region;
    * ``"aflq"``, augmented: the FLQ times ``log2(1 + SLQ_j)`` in the columns of sectors
      with ``SLQ_j > 1``, which are specialised and buy more locally.
    """
    reg = pd.Series(regional, dtype=float)
    nat = pd.Series(national, dtype=float)
    if set(reg.index) != set(nat.index):
        raise KeyError("regional and national sizes must name the same sectors")
    reg = reg.reindex(nat.index)
    if (reg < 0).any() or (nat <= 0).any():
        raise ValueError("sizes must be nonnegative, and positive nationally")
    if not 0 <= delta < 1:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    slq = (reg / reg.sum()) / (nat / nat.sum())
    s = slq.to_numpy()
    sectors = list(nat.index)
    if method == "slq":
        out = np.repeat(s[:, None], len(s), axis=1)
    elif method in ("cilq", "flq", "aflq"):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = s[:, None] / s[None, :]
        np.fill_diagonal(out, s)
        if method in ("flq", "aflq"):
            out = out * np.log2(1.0 + reg.sum() / nat.sum()) ** delta
        if method == "aflq":
            out = out * np.where(s > 1.0, np.log2(1.0 + s), 1.0)[None, :]
    else:
        raise ValueError(f"method must be 'slq', 'cilq', 'flq' or 'aflq', got {method!r}")
    return pd.DataFrame(out, index=sectors, columns=sectors)


def regional_coefficients(national: pd.DataFrame, quotients: pd.DataFrame) -> pd.DataFrame:
    """Regional input coefficients, ``a_ij min(1, LQ_ij)``: imports make up the rest.

    ``national`` holds the national technical coefficients, supplying sectors in rows and
    purchasing sectors in columns, labelled as ``quotients`` is. A quotient of one or more
    leaves the coefficient as it is; below one, the region buys that share locally.
    """
    if set(national.index) != set(quotients.index) or set(national.columns) != set(
            quotients.columns):
        raise KeyError("national coefficients and quotients must have the same labels")
    lq = quotients.reindex(index=national.index, columns=national.columns)
    return national * np.minimum(lq.to_numpy(dtype=float), 1.0)
