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

Mode B, a transformation between segments
-----------------------------------------
With ``ModelSpec.segment_mobility = {group: ω}``, each household's workforce is
a pool of :class:`cgelab.mobility.FactorMobility`, transformed across the
segments by a CET or a logit at the segments' wages. A flexible segment's wage
clears it, ``Σ_h S_hg = D_g``. A rigid segment hires what demand asks for, and
those it does not hire queue in it: ``U_g = Σ_h S_hg − D_g``, with the base-year
rate ``u_g0`` setting its base supply ``L_g0 / (1 − u_g0)``. There is no buffer.

Who the worker is
-----------------
Losing a formal job moves a person to the informal segment or to the queue
within the same household. In mode A each household has a fixed workforce in
the group, ``n_h = Σ_g s_hg L_g0 + q_h U0``, with ``s_hg`` its share of segment
``g`` in the SAM and ``q_h`` its share of the base year's rigid jobs. It holds the
share ``s_hg`` of every rigid and flexible segment's jobs, the share ``q_h`` of
the queue, and the rest of its workforce works in the informal segment. In mode
B the household supplies ``S_hg`` and a rigid segment hires each household's
supply in the same proportion, ``D_g / Σ_h S_hg``. At the base year both return
the SAM's ownership; away from it the income follows the people, which is the
rule of "Composition is not welfare": a household account keeps its members.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .mobility import FactorMobility

__all__ = ["LabourMarket", "SegmentGroup"]


@dataclass
class LabourMarket:
    """How one group of labour segments clears (``Closure.labour_markets``)."""

    #: Rigid segments, ``{factor: real wage}``, in base-year units (1 is the
    #: calibrated real wage; None means 1). The wage is that target times the
    #: purchaser-price index, and employment is what demand asks for at it.
    rigid: Mapping[str, float | None] = field(default_factory=dict)
    #: The segment that absorbs whoever the others do not hire: the informal
    #: one. Absent, the group's last segment. Perfect mobility (mode A) only.
    informal: str | None = None
    #: ``"none"``: a dual labour market, everyone not hired by a rigid segment
    #: works in the informal one. ``"harris_todaro"``: workers queue for rigid
    #: jobs until the informal wage equals the expected rigid wage. Mode A only.
    unemployment: str = "none"
    #: Unemployment in the base year. Under Harris–Todaro, one rate: the share
    #: of those in or queueing for rigid jobs, strictly between 0 and 1. Under
    #: a transformation between segments (mode B), the queue in each rigid
    #: segment as a share of its supply: one rate for every rigid segment, or
    #: ``{factor: rate}``. A dual market has none.
    unemployment_rate: float | Mapping[str, float] = 0.0

    def __post_init__(self):
        if self.unemployment not in ("none", "harris_todaro"):
            raise ValueError("LabourMarket.unemployment must be 'none' or 'harris_todaro', "
                             f"got {self.unemployment!r}")
        rates = (dict(self.unemployment_rate) if isinstance(self.unemployment_rate, Mapping)
                 else {None: self.unemployment_rate})
        for f, rate in rates.items():
            if not 0.0 <= float(rate) < 1.0:
                raise ValueError(f"LabourMarket.unemployment_rate must lie in [0, 1), got "
                                 f"{rate!r}" + (f" for {f!r}" if f else ""))
            if f is not None and f not in self.rigid:
                raise ValueError(f"LabourMarket.unemployment_rate names {f!r}, which is not "
                                 f"rigid: only a rigid segment has a queue")
        if self.unemployment == "harris_todaro":
            if not self.rigid:
                raise ValueError(
                    "a Harris–Todaro labour market needs a rigid segment: the queue is for "
                    "the jobs that pay the fixed wage")
            if isinstance(self.unemployment_rate, Mapping) or not float(
                    self.unemployment_rate) > 0.0:
                raise ValueError(
                    "LabourMarket.unemployment_rate must be one rate strictly between 0 and 1 "
                    f"under Harris–Todaro, got {self.unemployment_rate!r}. The SAM records "
                    "employment; the base year sits at this rate, and λ = 1 / (1 − rate) is "
                    "calibrated from it.")
        for f, target in self.rigid.items():
            if target is not None and not float(target) > 0:
                raise ValueError(f"LabourMarket.rigid[{f!r}] must be a positive real wage, "
                                 f"got {target!r}")
        if self.informal is not None and self.informal in self.rigid:
            raise ValueError(f"the informal segment {self.informal!r} cannot also be rigid: "
                             f"its wage is the one that clears")

    def rate_of(self, factor: str) -> float:
        """Base-year queue rate of a rigid segment, under a transformation between segments."""
        if factor not in self.rigid:
            return 0.0
        if isinstance(self.unemployment_rate, Mapping):
            return float(self.unemployment_rate.get(factor, 0.0))
        return float(self.unemployment_rate)


@dataclass
class SegmentGroup:
    """A calibrated group of labour segments."""

    name: str
    factors: list
    index: list               # factor indices, in the group's order
    base: np.ndarray          # base-year employment of each segment
    #: Transformation between segments (``ModelSpec.segment_mobility``): its
    #: elasticity and form. None: perfect mobility (mode A).
    omega: float | None = None
    form: str = "cet"


@dataclass
class GroupState:
    """One group in one period: who is rigid, the population, and its households."""

    group: SegmentGroup
    rigid: list               # factor indices
    target: np.ndarray        # real wage target of each rigid segment
    flexible: list            # mode A: factor indices sharing the informal wage
    informal: int             # mode A: the informal segment; -1 in mode B
    harris_todaro: bool
    lam: float
    U0: float
    rigid_base: float         # base-year employment of the rigid segments
    N: float                  # labour force: employment plus the queue
    shares: np.ndarray        # (n_hh, n_seg) household shares of each segment
    workforce: np.ndarray     # (n_hh,) n_h
    queue: np.ndarray         # (n_hh,) q_h, mode A
    #: Mode B: the transformation, with households that hold labour in the
    #: group as its pools (``active``), and the segments whose wage clears them.
    transform: FactorMobility | None = None
    active: np.ndarray | None = None
    clearing: list = field(default_factory=list)

    @property
    def hired(self) -> list:
        """Segments whose employment is what demand asks for."""
        return self.rigid + self.flexible

    def supplies(self, w: np.ndarray) -> np.ndarray:
        """Mode B: each household's supply to each segment, ``(n_hh, n_seg)``."""
        out = np.zeros_like(self.shares)
        S, _ = self.transform.supply(w[self.group.index], self.workforce[self.active])
        out[self.active] = S
        return out

    def households(self, paid: np.ndarray, U: float, supply=None) -> tuple:
        """Employment by household and segment, ``(n_hh, n_seg)``, and the queue by household.

        Affine in ``paid`` and ``U`` (mode A), or in ``paid`` at given supplies
        (mode B), which is what lets the solver find employment in closed form.
        """
        idx = self.group.index
        if self.transform is not None:
            e = supply.copy()
            for i in self.rigid:
                k = idx.index(i)
                total = float(supply[:, k].sum())
                e[:, k] = supply[:, k] * (paid[i] / total if total > 0 else 0.0)
            return e, (supply - e).sum(axis=1)
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
    mobility = dict(spec.segment_mobility or {})
    for name, members in (spec.labour_segments or {}).items():
        idx = [facs.index(f) for f in members]
        out[name] = SegmentGroup(
            name=name, factors=list(members), index=idx,
            base=np.asarray(fac_supply, float)[idx].copy(),
            omega=None if name not in mobility else float(mobility[name]),
            form=spec.mobility_form_of(name))
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
        rigid = [econ.factor(f) for f in grp.factors if f in mkt.rigid]
        target = np.array([1.0 if mkt.rigid[econ.factors[i]] is None
                           else float(mkt.rigid[econ.factors[i]]) for i in rigid])
        S = np.asarray(facshare, float)[:, grp.index]
        L0 = grp.base
        rigid_k = [grp.index.index(i) for i in rigid]
        L_R0 = float(L0[rigid_k].sum())
        if name in forces:
            N_given = float(forces[name])
            if not N_given > 0:
                raise ValueError(f"Scenario.labour_force[{name!r}] must be positive, "
                                 f"got {N_given}")
        else:
            N_given = None
        scale = float(np.sum(supply[grp.index])) / float(L0.sum())

        if grp.omega is not None:
            states.append(_transformed(econ, name, grp, mkt, rigid, rigid_k, target, S, L0,
                                       L_R0, N_given, scale))
            continue

        dual_with_rate = mkt.unemployment == "none" and (
            isinstance(mkt.unemployment_rate, Mapping) or float(mkt.unemployment_rate) != 0.0)
        if dual_with_rate:
            raise ValueError(
                f"group {name!r}: a dual labour market has no unemployment, so "
                f"LabourMarket.unemployment_rate is read only under 'harris_todaro' or with "
                f"ModelSpec.segment_mobility, where each rigid segment has a queue.")
        informal = mkt.informal if mkt.informal is not None else grp.factors[-1]
        if informal in mkt.rigid:
            raise ValueError(f"the informal segment {informal!r} of group {name!r} cannot be "
                             f"rigid; name another one with LabourMarket.informal")
        b = econ.factor(informal)
        flexible = [i for i in grp.index if i not in rigid and i != b]
        if mkt.unemployment == "harris_todaro" and flexible:
            raise ValueError(
                f"group {name!r}: under Harris–Todaro every segment is rigid or the informal "
                f"one, so it is clear who queues; {[econ.factors[i] for i in flexible]} are "
                f"neither. Make them rigid, or use unemployment='none'.")
        u0 = float(mkt.unemployment_rate)
        U0 = u0 / (1.0 - u0) * L_R0 if mkt.unemployment == "harris_todaro" else 0.0
        lam = 1.0 / (1.0 - u0)
        queue = (S[:, rigid_k] @ L0[rigid_k] / L_R0) if L_R0 > 0 else np.zeros(S.shape[0])
        N0 = float(L0.sum()) + U0
        N = N_given if N_given is not None else N0 * scale
        workforce = (S @ L0 + queue * U0) * (N / N0)
        states.append(GroupState(
            group=grp, rigid=rigid, target=target, flexible=flexible, informal=b,
            harris_todaro=mkt.unemployment == "harris_todaro", lam=lam, U0=U0,
            rigid_base=max(L_R0, 1.0), N=N,
            shares=S, workforce=workforce, queue=queue))
    return states


def _transformed(econ, name, grp, mkt, rigid, rigid_k, target, S, L0, L_R0, N_given, scale):
    """Mode B: each household's workforce a pool of a CET or logit across the segments."""
    if mkt.informal is not None:
        raise ValueError(
            f"group {name!r} has a transformation between segments (segment_mobility), so "
            f"nobody is absorbed by an informal segment: each flexible segment clears and "
            f"each rigid one has its own queue. Leave LabourMarket.informal out.")
    if mkt.unemployment == "harris_todaro":
        raise ValueError(
            f"group {name!r}: Harris–Todaro is a rule for perfect mobility between segments. "
            f"With segment_mobility, each rigid segment's queue is the supply it does not "
            f"hire; set its base-year rate with LabourMarket.unemployment_rate.")
    rates = np.array([mkt.rate_of(f) for f in grp.factors])
    S0 = L0 / (1.0 - rates)                       # base supply, the queue included
    Sh0 = S * S0[None, :]
    n0 = Sh0.sum(axis=1)
    active = n0 > 0
    N0 = float(S0.sum())
    N = N_given if N_given is not None else N0 * scale
    workforce = n0 * (N / N0)
    transform = FactorMobility(
        factor=name, index=-1, form=grp.form, omega=grp.omega,
        users=np.ones(len(grp.index), dtype=bool),
        pools=[econ.hhs[h] for h in np.where(active)[0]],
        delta=Sh0[active] / n0[active, None], endowment0=n0[active])
    clearing = [i for i in grp.index if i not in rigid]
    return GroupState(
        group=grp, rigid=rigid, target=target, flexible=[], informal=-1,
        harris_todaro=False, lam=1.0, U0=float((S0 - L0).sum()),
        rigid_base=max(L_R0, 1.0), N=N, shares=S, workforce=workforce,
        queue=np.zeros(S.shape[0]), transform=transform, active=active, clearing=clearing)
