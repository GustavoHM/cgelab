"""Segmented labour: several labour accounts that share one population of workers.

A SAM often pays labour through more than one account: formal and informal,
or covered and uncovered. Each is a factor the value-added nest demands on its
own. A group of segments (``ModelSpec.labour_segments``) says that the same
people supply all of them, and a labour market (``Closure.labour_markets``)
says how each segment's wage is set and where whoever a rigid segment does not
hire goes.

Mode A, perfect mobility between segments
-----------------------------------------
Every segment that is not rigid earns one common wage, the one that clears the
informal (buffer) segment. A rigid segment pays a fixed real wage, a minimum
wage or a formal floor, and hires what demand asks for at it::

    w_g = ŵ_g P                              g rigid
    E_g = D_g(w)                             g rigid or flexible
    E_b = N − Σ_{g≠b} E_g − U                the informal segment b absorbs the rest
    U   = 0                                  dual labour market
    w_b = λ Σ_R w_g E_g / (Σ_R E_g + U)      Harris–Todaro, λ = 1 / (1 − u0)

Under Harris–Todaro the informal wage equals the wage expected by queueing for a
rigid job, and λ makes the base year, with its unemployment rate u0, reproduce
the SAM.

Who the worker is
-----------------
Losing a formal job moves a person to the informal segment or to the queue
within the same household. Each household has a fixed workforce in the group,
``n_h = Σ_g s_hg L_g0 + q_h U0``, with ``s_hg`` its share of segment ``g`` in
the SAM and ``q_h`` its share of the base year's rigid jobs. It holds the share
``s_hg`` of every rigid and flexible segment's jobs, the share ``q_h`` of the
queue, and the rest of its workforce works in the informal segment. At the
base year that returns the SAM's ownership; away from it the informal income
follows the people who lost formal jobs, instead of going to the households
that were already informal. That is the rule of "Composition is not welfare":
a household account keeps its members.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

__all__ = ["LabourMarket", "SegmentGroup"]


@dataclass
class LabourMarket:
    """How one group of labour segments clears (``Closure.labour_markets``)."""

    #: Rigid segments, ``{factor: real wage}``, in base-year units (1 is the
    #: calibrated real wage; None means 1). The wage is that target times the
    #: purchaser-price index, and employment is what demand asks for at it.
    rigid: Mapping[str, float | None] = field(default_factory=dict)
    #: The segment that absorbs whoever the others do not hire: the informal
    #: one. Absent, the group's last segment.
    informal: str | None = None
    #: ``"none"``: a dual labour market, everyone not hired by a rigid segment
    #: works in the informal one. ``"harris_todaro"``: workers queue for rigid
    #: jobs until the informal wage equals the expected rigid wage.
    unemployment: str = "none"
    #: Unemployment in the base year, as a share of those in or queueing for
    #: rigid jobs. Harris–Todaro only, and then strictly between 0 and 1.
    unemployment_rate: float = 0.0

    def __post_init__(self):
        if self.unemployment not in ("none", "harris_todaro"):
            raise ValueError("LabourMarket.unemployment must be 'none' or 'harris_todaro', "
                             f"got {self.unemployment!r}")
        rate = float(self.unemployment_rate)
        if self.unemployment == "harris_todaro":
            if not self.rigid:
                raise ValueError(
                    "a Harris–Todaro labour market needs a rigid segment: the queue is for "
                    "the jobs that pay the fixed wage")
            if not 0.0 < rate < 1.0:
                raise ValueError(
                    "LabourMarket.unemployment_rate must lie strictly between 0 and 1 under "
                    f"Harris–Todaro, got {rate}. The SAM records employment; the base year "
                    "sits at this rate, and λ = 1 / (1 − rate) is calibrated from it.")
        elif rate != 0.0:
            raise ValueError(
                "LabourMarket.unemployment_rate is read only under 'harris_todaro'. A dual "
                "labour market has no unemployment: the informal segment absorbs everyone.")
        for f, target in self.rigid.items():
            if target is not None and not float(target) > 0:
                raise ValueError(f"LabourMarket.rigid[{f!r}] must be a positive real wage, "
                                 f"got {target!r}")
        if self.informal is not None and self.informal in self.rigid:
            raise ValueError(f"the informal segment {self.informal!r} cannot also be rigid: "
                             f"its wage is the one that clears")


@dataclass
class SegmentGroup:
    """A calibrated group of labour segments."""

    name: str
    factors: list
    index: list               # factor indices, in the group's order
    base: np.ndarray          # base-year employment of each segment


@dataclass
class GroupState:
    """One group in one period: who is rigid, the population, and its households."""

    group: SegmentGroup
    rigid: list               # factor indices
    target: np.ndarray        # real wage target of each rigid segment
    flexible: list            # factor indices sharing the informal wage, informal excluded
    informal: int
    harris_todaro: bool
    lam: float
    U0: float
    rigid_base: float         # base-year employment of the rigid segments
    N: float                  # labour force: employment plus the queue
    shares: np.ndarray        # (n_hh, n_seg) household shares of each segment
    workforce: np.ndarray     # (n_hh,) n_h
    queue: np.ndarray         # (n_hh,) q_h

    @property
    def hired(self) -> list:
        """Segments whose employment is what demand asks for."""
        return self.rigid + self.flexible

    def households(self, paid: np.ndarray, U: float) -> tuple:
        """Employment by household and segment, ``(n_hh, n_seg)``, and the queue by household.

        Affine in ``paid`` and ``U``, which is what lets the solver find
        employment in closed form.
        """
        idx = self.group.index
        e = np.zeros_like(self.shares)
        hired = set(self.hired)
        for k, i in enumerate(idx):
            if i in hired:
                e[:, k] = self.shares[:, k] * paid[i]
        u_h = self.queue * U
        k_b = idx.index(self.informal)
        e[:, k_b] = self.workforce - e.sum(axis=1) - u_h
        return e, u_h


def build_segments(spec, facs, fac_supply) -> dict:
    out = {}
    for name, members in (spec.labour_segments or {}).items():
        idx = [facs.index(f) for f in members]
        out[name] = SegmentGroup(name=name, factors=list(members), index=idx,
                                 base=np.asarray(fac_supply, float)[idx].copy())
    return out


def period_groups(econ, closure, scenario, supply, facshare) -> list:
    """The state of every segment group this period."""
    markets = dict(closure.labour_markets or {})
    stray = [g for g in markets if g not in econ.segments]
    if stray:
        raise ValueError(f"Closure.labour_markets names {stray}, which are not groups of "
                         f"ModelSpec.labour_segments ({list(econ.segments)})")
    forces = dict(scenario.labour_force or {})
    stray = [g for g in forces if g not in econ.segments]
    if stray:
        raise ValueError(f"Scenario.labour_force names {stray}, which are not groups of "
                         f"ModelSpec.labour_segments ({list(econ.segments)})")
    states = []
    for name, grp in econ.segments.items():
        mkt = markets.get(name, LabourMarket())
        unknown = [f for f in mkt.rigid if f not in grp.factors]
        if unknown or (mkt.informal is not None and mkt.informal not in grp.factors):
            raise ValueError(
                f"Closure.labour_markets[{name!r}] names segments that are not in the group "
                f"{grp.factors}: {unknown or [mkt.informal]}")
        informal = mkt.informal if mkt.informal is not None else grp.factors[-1]
        if informal in mkt.rigid:
            raise ValueError(f"the informal segment {informal!r} of group {name!r} cannot be "
                             f"rigid; name another one with LabourMarket.informal")
        rigid = [econ.factor(f) for f in grp.factors if f in mkt.rigid]
        target = np.array([1.0 if mkt.rigid[econ.factors[i]] is None
                           else float(mkt.rigid[econ.factors[i]]) for i in rigid])
        b = econ.factor(informal)
        flexible = [i for i in grp.index if i not in rigid and i != b]
        if mkt.unemployment == "harris_todaro" and flexible:
            raise ValueError(
                f"group {name!r}: under Harris–Todaro every segment is rigid or the informal "
                f"one, so it is clear who queues; {[econ.factors[i] for i in flexible]} are "
                f"neither. Make them rigid, or use unemployment='none'.")
        L0 = grp.base
        rigid_k = [grp.index.index(i) for i in rigid]
        L_R0 = float(L0[rigid_k].sum())
        u0 = float(mkt.unemployment_rate)
        U0 = u0 / (1.0 - u0) * L_R0 if mkt.unemployment == "harris_todaro" else 0.0
        lam = 1.0 / (1.0 - u0)
        S = np.asarray(facshare, float)[:, grp.index]
        queue = (S[:, rigid_k] @ L0[rigid_k] / L_R0) if L_R0 > 0 else np.zeros(S.shape[0])
        N0 = float(L0.sum()) + U0
        if name in forces:
            N = float(forces[name])
            if not N > 0:
                raise ValueError(f"Scenario.labour_force[{name!r}] must be positive, got {N}")
        else:
            N = N0 * float(np.sum(supply[grp.index])) / float(L0.sum())
        workforce = (S @ L0 + queue * U0) * (N / N0)
        states.append(GroupState(
            group=grp, rigid=rigid, target=target, flexible=flexible, informal=b,
            harris_todaro=mkt.unemployment == "harris_todaro", lam=lam, U0=U0,
            rigid_base=max(L_R0, 1.0), N=N,
            shares=S, workforce=workforce, queue=queue))
    return states
