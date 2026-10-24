"""Imperfect factor mobility: a factor's endowment transformed across activities.

A perfectly mobile factor earns one price everywhere, and a sector-specific one
is a separate factor. Between the two, the endowment of a factor is split among
the activities that use it by a transformation frontier with elasticity ω, and
each activity pays its own price. ω = 0 holds the base-year allocation; a large
ω approaches one price.

The endowment can come in pools. Each pool is a group of owners of the factor
(an occupation, a region of origin, a set of neighbouring sectors) that supplies
the activities it reaches in the base year, and only those. A proximity matrix
is that pool-by-activity table of base-year use: a pool hires out mostly to the
activities near it, rather than every activity drawing on one national pool.
Without pools there is one, the whole endowment.

Two forms
---------
``"cet"`` is the transformation frontier of GTAP's sluggish endowments and of
ORANI's occupational transformation. Pool ``o`` with endowment ``L_o`` and
base-year shares ``δ_oa`` has the price index and supplies::

    W_o  = [ Σ_a δ_oa w_a^(1+ω) ]^(1/(1+ω))
    S_oa = δ_oa L_o (w_a / W_o)^ω

The frontier is homogeneous of degree one, so the pool's income is
``Σ_a w_a S_oa = W_o L_o`` exactly: the income accounting closes by identity.
The sum of the supplies in physical units, ``Σ_a S_oa``, is not ``L_o`` away
from equal prices. Moving between activities costs efficiency.

``"logit"`` keeps the head count: ``S_oa = L_o δ_oa w_a^ω / Σ_b δ_ob w_b^ω``, so
``Σ_a S_oa = L_o`` whatever the prices. The pool's price is then its income over
its head count, the average wage it earns, and it has no dual frontier.

To first order around equal prices the two forms are the same supply function.
They part at second order, which is what a test can check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

__all__ = ["FactorMobility"]


@dataclass
class FactorMobility:
    """The calibrated transformation of one factor's endowment across activities."""

    factor: str
    index: int
    form: str                 # "cet" | "logit"
    omega: float
    users: np.ndarray         # (n_act,) bool: activities that use the factor in the base year
    pools: list               # pool names
    delta: np.ndarray         # (n_pool, n_act) base-year shares of each pool, rows sum to one
    endowment0: np.ndarray    # (n_pool,) base-year endowment of each pool
    #: Allow destinations absent in the base year (regional migration, not sectoral pools).
    allow_new: bool = False

    @property
    def n_pool(self) -> int:
        return len(self.pools)

    def supply(self, wage: np.ndarray, endowment: np.ndarray) -> tuple:
        """Supply to each activity, by pool, and each pool's price.

        ``wage`` is the factor's price in each activity, ``(n_act,)``; entries
        outside ``users`` are not read. Returns ``(S, W)`` with ``S`` of shape
        ``(n_pool, n_act)`` and ``W`` of shape ``(n_pool,)``. Pool income is
        ``W * endowment`` in both forms.
        """
        u = self.users
        logw = np.zeros(self.delta.shape[1])
        logw[u] = np.log(np.asarray(wage, dtype=float)[u])
        reach = (np.ones_like(self.delta, dtype=bool) if self.allow_new
                 else self.delta > 0)
        # In logs: a large ω on a wage a little above one would overflow w^(1+ω).
        with np.errstate(divide="ignore"):
            logd = (np.log(np.maximum(self.delta, 1e-300)) if self.allow_new
                    else np.where(reach, np.log(np.where(reach, self.delta, 1.0)), -np.inf))
        om = self.omega
        endowment = np.asarray(endowment, dtype=float)
        if self.form == "cet":
            a = logd + (1.0 + om) * logw[None, :]
            top = a.max(axis=1, keepdims=True)
            logW = (top[:, 0] + np.log(np.exp(a - top).sum(axis=1))) / (1.0 + om)
            S = np.where(reach, np.exp(logd + om * (logw[None, :] - logW[:, None])), 0.0)
            S = S * endowment[:, None]
            return S, np.exp(logW)
        a = logd + om * logw[None, :]
        top = a.max(axis=1, keepdims=True)
        share = np.where(reach, np.exp(a - top), 0.0)
        share = share / share.sum(axis=1, keepdims=True)
        S = share * endowment[:, None]
        W = (share * np.exp(logw)[None, :]).sum(axis=1)
        return S, W


def build_mobility(spec, facs, acts, base_use, factor_pools) -> dict:
    """Calibrate every factor named in ``spec.factor_mobility``.

    ``base_use`` is ``(n_fac, n_act)``, the factor use the base year clears at
    unit prices. ``factor_pools`` maps a factor to a pool-by-activity table of
    that use; its columns must add up to the factor's use in each activity.
    """
    import pandas as pd

    mobility = spec.factor_mobility or {}
    pools_in = dict(factor_pools or {})
    stray = [f for f in pools_in if f not in mobility and f not in (spec.factor_migration or {})]
    if stray:
        raise ValueError(
            f"calibrate(factor_pools=) names {stray}, which ModelSpec.factor_mobility does "
            f"not list. Pools only exist for a factor that is not perfectly mobile.")
    out = {}
    for f, omega in mobility.items():
        if f not in facs:
            raise KeyError(f"ModelSpec.factor_mobility names {f!r}, which is not among "
                           f"the factors {list(facs)}")
        i = facs.index(f)
        use = np.asarray(base_use[i], dtype=float)
        if np.any(use < -1e-12):
            bad = [acts[j] for j in np.where(use < -1e-12)[0]]
            raise ValueError(
                f"factor {f!r} has negative base-year use in {bad}. A transformation "
                f"frontier cannot supply a negative quantity; leave {f!r} perfectly mobile "
                f"or treat the activity's value added first.")
        users = use > 1e-12
        if not users.any():
            raise ValueError(f"factor {f!r} is used by no activity in the base year")
        if f in pools_in:
            table = pools_in[f]
            if isinstance(table, str) and table == "activities":
                # A pool for each activity that uses the factor, holding what it
                # uses: sector-specific capital, or labour tied to its sector.
                keep = [a for a, u in zip(acts, users) if u]
                table = pd.DataFrame(np.diag(use[users]), index=keep, columns=keep)
            if not isinstance(table, pd.DataFrame):
                raise TypeError(f"factor_pools[{f!r}] must be a DataFrame, pools by "
                                f"activities, or \"activities\" for a pool per activity")
            unknown = [c for c in table.columns if c not in acts]
            if unknown:
                raise KeyError(f"factor_pools[{f!r}] has columns that are not activities: "
                               f"{unknown}")
            F = np.zeros((len(table.index), len(acts)))
            for c in table.columns:
                F[:, acts.index(c)] = table[c].to_numpy(dtype=float)
            if np.any(F < 0):
                raise ValueError(f"factor_pools[{f!r}] has negative entries")
            off = [acts[j] for j in np.where((F.sum(axis=0) > 0) & ~users)[0]]
            if off:
                raise ValueError(
                    f"factor_pools[{f!r}] supplies {off}, which use no {f!r} in the base year")
            gap = np.abs(F.sum(axis=0) - np.where(users, use, 0.0))
            bad = [acts[j] for j in np.where(gap > 1e-8 * np.maximum(np.abs(use), 1.0))[0]]
            if bad:
                raise ValueError(
                    f"factor_pools[{f!r}] does not add up to the base-year use of {f!r} in "
                    f"{bad}: every column must sum to what the SAM pays the factor there, "
                    f"or the base year will not clear.")
            empty = [str(p) for p, r in zip(table.index, F.sum(axis=1)) if r <= 0]
            if empty:
                raise ValueError(f"factor_pools[{f!r}]: pools {empty} supply nothing")
            names = [str(p) for p in table.index]
        else:
            F = np.where(users, use, 0.0)[None, :]
            names = [f]
        endow = F.sum(axis=1)
        out[i] = FactorMobility(
            factor=f, index=i, form=spec.mobility_form_of(f), omega=float(omega),
            users=users, pools=names, delta=F / endow[:, None], endowment0=endow)
    return out


def period_endowments(econ, scenario) -> tuple:
    """Factor supplies and pool endowments for one period: ``(supply, {index: endowment})``.

    A factor whose pools ``Scenario.pool_supply`` sets, and whose supply
    ``Scenario.factor_supply`` leaves unset, has the pools' sum as its supply.
    """
    supply = (econ.fac_supply if scenario.factor_supply is None
              else np.asarray(scenario.factor_supply, float)).copy()
    given = dict(scenario.pool_supply or {})
    stray = [f for f in given if f not in econ.factors or econ.factor(f) not in econ.mobility]
    if stray:
        raise ValueError(f"Scenario.pool_supply names {stray}, which are not imperfectly "
                         f"mobile factors of this economy")
    if scenario.factor_supply is None:
        for f, v in given.items():
            supply[econ.factor(f)] = float(np.sum(v))
    endow = {i: pool_endowment(m, float(supply[i]), float(econ.fac_supply[i]), given)
             for i, m in econ.mobility.items()}
    return supply, endow


def pool_endowment(mob: FactorMobility, total: float, base_total: float,
                   given: Mapping | None) -> np.ndarray:
    """Endowment of each pool this period.

    Absent from ``Scenario.pool_supply``, the pools scale with the factor's
    supply, so a growth path that raises labour raises every pool alike.
    """
    if given is not None and mob.factor in given:
        v = np.asarray(given[mob.factor], dtype=float)
        if v.shape != (mob.n_pool,):
            raise ValueError(
                f"Scenario.pool_supply[{mob.factor!r}] must have one entry per pool "
                f"({mob.pools}), got shape {v.shape}")
        if np.any(v < 0):
            raise ValueError(f"Scenario.pool_supply[{mob.factor!r}] has a negative endowment")
        if abs(float(v.sum()) - total) > 1e-10 * max(abs(total), 1.0):
            raise ValueError(
                f"Scenario.pool_supply[{mob.factor!r}] sums to {float(v.sum()):.10g}, but the "
                f"factor's supply is {total:.10g}. Set Scenario.factor_supply to the same "
                f"total, or leave one of them out.")
        return v
    return mob.endowment0 * (total / base_total if base_total else 0.0)
