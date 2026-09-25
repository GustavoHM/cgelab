"""Welfare measurement: equivalent variation, by account and by group.

Two cautions are built into the interface rather than left to the user.

**Composition.** An account is not the same population in two scenarios whenever
households MOVE between accounts -- adopters and non-adopters, formal and
informal. Per-account real income then carries that size difference and reports
it as welfare. :func:`equivalent_variation` computes both, and names the
per-account column so that no call site can take it for the distributional
object by accident; :func:`ev_by_group` sums real incomes WITHIN a group before
dividing, so the set of households summed over is the same in both scenarios and
the ratio does not depend on how many migrated.

**Yardstick.** ``ev`` values the change at the reference scenario's own prices;
``ev_base`` values it at base-year prices. They differ by the price drift
accumulated over the horizon, which is a level effect that does NOT shrink as
the policy shock shrinks. Reconciling a sum of EVs against real absorption, which
is itself measured at base-year prices, needs ``ev_base``; comparing like with
like is not a detail.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["equivalent_variation", "ev_by_group", "compensating_variation"]


def _panel(path) -> pd.DataFrame:
    df = path.households().copy()
    # Direct tax is not spendable. The column is absent from panels built before
    # direct taxes existed, and zero whenever the SAM has none.
    tax = df["direct_tax"] if "direct_tax" in df else 0.0
    df["welfare_income"] = df["income"] - df["direct_outlay"] - tax
    df["real_income"] = df["welfare_income"] / df["price_index"]
    return df


def equivalent_variation(reference, alternative, *, base_price_index=None) -> pd.DataFrame:
    """EV of moving from ``reference`` to ``alternative``, per year and account.

    Both arguments are :class:`~cgekit.dynamics.PathResult` objects. ``ev`` is
    valued at the reference scenario's prices, ``ev_base`` at base-year prices.
    """
    ref = _panel(reference)
    alt = _panel(alternative)
    m = ref.merge(alt, on=["year", "account"], suffixes=("_ref", "_alt"))
    if m.empty:
        raise ValueError("the two paths share no (year, account) pair")

    base_idx = 1.0 if base_price_index is None else np.asarray(base_price_index)
    m["ev"] = m.welfare_income_ref * (m.real_income_alt / m.real_income_ref - 1.0)
    m["ev_base"] = (m.real_income_alt - m.real_income_ref) * base_idx
    # Named to say what it is. A per-ACCOUNT percentage is not a per-household
    # one whenever the accounts change size between scenarios.
    m["ev_pct_account"] = 100.0 * (m.real_income_alt / m.real_income_ref - 1.0)
    m["wedge_paid_change"] = m.wedge_paid_alt - m.wedge_paid_ref
    m["direct_outlay_change"] = m.direct_outlay_alt - m.direct_outlay_ref
    return m[["year", "account", "ev", "ev_base", "ev_pct_account",
              "welfare_income_ref", "welfare_income_alt",
              "real_income_ref", "real_income_alt",
              "wedge_paid_change", "direct_outlay_change"]]


def ev_by_group(ev: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    """Aggregate EV within groups of accounts, immune to composition.

    Real incomes are summed across the accounts of a group BEFORE the ratio is
    taken, so migration between accounts inside a group cancels. Every account in
    ``ev`` must appear in ``groups``: silently dropping one would understate the
    group and the error would look like a result.
    """
    missing = sorted(set(ev.account) - set(groups))
    if missing:
        raise KeyError(f"no group given for accounts {missing}")
    df = ev.copy()
    df["group"] = df.account.map(groups)
    g = df.groupby(["year", "group"], as_index=False)[
        ["real_income_ref", "real_income_alt", "welfare_income_ref", "ev_base"]].sum()
    g["ev_pct"] = 100.0 * (g.real_income_alt / g.real_income_ref - 1.0)
    g["ev"] = g.welfare_income_ref * (g.real_income_alt / g.real_income_ref - 1.0)
    return g[["year", "group", "ev", "ev_base", "ev_pct",
              "real_income_ref", "real_income_alt"]]


def compensating_variation(reference, alternative) -> pd.DataFrame:
    """CV of the same move: the change valued at the ALTERNATIVE's prices.

    Reported alongside EV when the price change is large enough for the two to
    differ materially. They coincide to first order, and quoting only one while
    calling it "the welfare change" hides a choice of reference prices.
    """
    ref = _panel(reference)
    alt = _panel(alternative)
    m = ref.merge(alt, on=["year", "account"], suffixes=("_ref", "_alt"))
    m["cv"] = m.welfare_income_alt * (1.0 - m.real_income_ref / m.real_income_alt)
    return m[["year", "account", "cv"]]
