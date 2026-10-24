"""Social accounting matrix: container, classification, balance check, balancing.

A SAM is square and its row and column totals must agree account by account: what
an account receives it spends. Everything a CGE model knows about the base year
comes from that table, so an imbalance is never a rounding matter to be smoothed
over -- it is a statement that the base year does not add up, and a model
calibrated on it produces confident nonsense that no output makes visible.

The balancing routine here is therefore deliberately reluctant. It exists, it
works, and the docstring says plainly when NOT to use it: a residual large enough
to see is evidence about how the table was built, and closing it before
diagnosing it destroys that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SAM", "BalanceError", "sam_distance"]


def sam_distance(sam: "SAM", other: "SAM") -> float:
    """Kullback--Leibler divergence from ``other`` to ``sam``, ``D_KL(sam || other)``.

    The two matrices are aligned by account name and their nonnegative flows are
    normalised to sum to one. In a forecast comparison, call this as
    ``sam_distance(observed, predicted)``. A zero in ``other`` where ``sam`` has
    positive mass gives infinity. Negative flows have no probability
    interpretation and are refused rather than shifted or silently discarded.
    """
    accounts = sorted(set(sam.accounts) | set(other.accounts))
    p = sam.matrix(accounts).to_numpy(dtype=float).ravel()
    q = other.matrix(accounts).to_numpy(dtype=float).ravel()
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("sam_distance needs finite SAM entries")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("Kullback--Leibler distance is undefined for negative SAM entries")
    sp, sq = p.sum(), q.sum()
    if not sp > 0 or not sq > 0:
        raise ValueError("sam_distance needs two SAMs with positive total flow")
    p, q = p / sp, q / sq
    positive = p > 0
    if (q[positive] == 0).any():
        return float("inf")
    terms = p[positive] * np.log(p[positive] / q[positive])
    return float(max(0.0, terms.sum()))


class BalanceError(ValueError):
    """Raised when a SAM does not balance to the requested tolerance."""


@dataclass
class SAM:
    """A square social accounting matrix, held in long form.

    ``frame`` has columns ``row``, ``col``, ``value``: the payment FROM the column
    account TO the row account, the convention in which a column is an account's
    outlays and a row its receipts.
    """

    frame: pd.DataFrame
    name: str = "sam"
    _accounts: list | None = field(default=None, repr=False)
    #: Where each block came from, when the SAM was assembled from its sources
    #: (:func:`cgelab.build.sut_to_sam`): one row per block, with its source,
    #: what it is, its rows and columns, total and number of cells. Operations
    #: that return a new SAM do not carry it over.
    provenance: pd.DataFrame | None = field(default=None, repr=False)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_long(cls, df: pd.DataFrame, *, row: str = "row", col: str = "col",
                  value: str = "value", name: str = "sam") -> "SAM":
        missing = [c for c in (row, col, value) if c not in df.columns]
        if missing:
            raise KeyError(f"columns {missing} not in the frame; got {list(df.columns)}")
        out = df[[row, col, value]].copy()
        out.columns = ["row", "col", "value"]
        out["value"] = out["value"].astype(float)
        out = out.groupby(["row", "col"], as_index=False, sort=False)["value"].sum()
        return cls(frame=out, name=name)

    @classmethod
    def from_square(cls, df: pd.DataFrame, name: str = "sam") -> "SAM":
        """Build from a square frame indexed by account name on both axes."""
        stacked = df.stack()
        long = stacked.reset_index()
        long.columns = ["row", "col", "value"]
        return cls.from_long(long[long.value != 0.0], name=name)

    @classmethod
    def read(cls, path, name: str | None = None, **kwargs) -> "SAM":
        """Read a ``.parquet`` or ``.csv`` in long form."""
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        return cls.from_long(df, name=name or path.stem, **kwargs)

    # -- structure -------------------------------------------------------
    @property
    def accounts(self) -> list:
        if self._accounts is None:
            self._accounts = sorted(set(self.frame.row) | set(self.frame.col))
        return self._accounts

    def matrix(self, accounts=None) -> pd.DataFrame:
        """Dense square representation, accounts in a fixed order on both axes."""
        acc = list(accounts) if accounts is not None else self.accounts
        idx = {a: i for i, a in enumerate(acc)}
        M = np.zeros((len(acc), len(acc)))
        for r, c, v in self.frame.itertuples(index=False):
            if r in idx and c in idx:
                M[idx[r], idx[c]] += v
        return pd.DataFrame(M, index=acc, columns=acc)

    def block(self, rows, cols) -> np.ndarray:
        """Sub-matrix for two ordered lists of accounts, as a plain array."""
        rows, cols = list(rows), list(cols)
        ri = {a: i for i, a in enumerate(rows)}
        ci = {a: i for i, a in enumerate(cols)}
        out = np.zeros((len(rows), len(cols)))
        sub = self.frame[self.frame.row.isin(ri) & self.frame.col.isin(ci)]
        for r, c, v in sub.itertuples(index=False):
            out[ri[r], ci[c]] += v
        return out

    def receipts(self) -> pd.Series:
        return self.frame.groupby("row").value.sum().reindex(self.accounts).fillna(0.0)

    def outlays(self) -> pd.Series:
        return self.frame.groupby("col").value.sum().reindex(self.accounts).fillna(0.0)

    @property
    def total(self) -> float:
        return float(self.frame.value.abs().sum())

    # -- balance ---------------------------------------------------------
    def imbalance(self) -> pd.Series:
        """Receipts minus outlays, account by account."""
        return (self.receipts() - self.outlays()).rename("imbalance")

    def check_balance(self, rel_tol: float = 1e-9, abs_tol: float = 1.0,
                      raise_on_fail: bool = True) -> dict:
        """Verify that every account balances.

        The test is relative to the SAM total with an absolute floor, so an
        account of negligible size is not held to an impossible standard while a
        large one cannot hide a large gap behind a small ratio.
        """
        gap = self.imbalance()
        limit = max(rel_tol * self.total, abs_tol)
        worst = gap.abs().max() if len(gap) else 0.0
        offenders = gap[gap.abs() > limit].sort_values(key=np.abs, ascending=False)
        ok = len(offenders) == 0
        if not ok and raise_on_fail:
            head = ", ".join(f"{k}: {v:,.1f}" for k, v in offenders.head(8).items())
            raise BalanceError(
                f"{len(offenders)} account(s) out of balance beyond {limit:,.3g} "
                f"({head}). Diagnose the construction before balancing: a residual "
                f"this size is evidence about how the table was built.")
        return dict(ok=ok, worst=float(worst), limit=float(limit),
                    offenders=offenders, imbalance=gap)

    def balance(self, *, rel_tol: float = 1e-9, abs_tol: float = 1.0,
                max_iter: int = 200, name: str | None = None) -> "SAM":
        """Sign-preserving RAS.

        Standard RAS is multiplicative and breaks on negative entries, and real
        SAMs have them -- government dissaving, destocking, a subsidy exceeding
        the production tax, a negative operating surplus in a year a sector
        collapsed. Scaling is applied to MAGNITUDES with signs held fixed, and
        each account's target is the midpoint of its receipts and outlays.

        Use this only after the residual has been diagnosed. It is the last
        resort, not the first response.
        """
        df = self.frame.copy()
        acc = self.accounts
        limit = max(rel_tol * self.total, abs_tol)
        for it in range(max_iter):
            rec = df.groupby("row").value.sum().reindex(acc).fillna(0.0)
            out = df.groupby("col").value.sum().reindex(acc).fillna(0.0)
            gap = rec - out
            if gap.abs().max() <= limit:
                return SAM(frame=df.reset_index(drop=True), name=name or f"{self.name}-balanced")
            target = 0.5 * (rec + out)
            mag_r = df.assign(a=df.value.abs()).groupby("row").a.sum().reindex(acc).fillna(0.0)
            mag_c = df.assign(a=df.value.abs()).groupby("col").a.sum().reindex(acc).fillna(0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.where(mag_r > 0, np.abs(target) / np.maximum(mag_r, 1e-12), 1.0)
                fc = np.where(mag_c > 0, np.abs(target) / np.maximum(mag_c, 1e-12), 1.0)
            fr = pd.Series(np.sqrt(np.clip(fr, 1e-6, 1e6)), index=acc)
            fc = pd.Series(np.sqrt(np.clip(fc, 1e-6, 1e6)), index=acc)
            df["value"] = df.value * df.row.map(fr).to_numpy() * df.col.map(fc).to_numpy()
        raise BalanceError(
            f"sign-preserving RAS did not converge in {max_iter} iterations "
            f"(worst gap {gap.abs().max():,.3g} against a limit of {limit:,.3g}). "
            f"Writing a SAM that merely looks balanced would be worse than failing.")

    def __repr__(self) -> str:
        return (f"SAM({self.name!r}, {len(self.accounts)} accounts, "
                f"{len(self.frame)} entries, total {self.total:,.0f})")
