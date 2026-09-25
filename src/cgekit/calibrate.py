"""Calibrate a CGE economy from a balanced SAM.

Every coefficient the solver uses is derived here, once, and the derivation is
where the units are decided. Two of those decisions are silent when wrong and
expensive to find later, so both are handled explicitly and documented at the
point of use:

* **Purchaser versus basic prices.** A SAM records intermediate and final
  purchases INCLUSIVE of product taxes. Quantities in the model are in base-year
  basic-price units, so those flows are divided by ``1 + t`` to convert them.
  Skipping it makes the tax appear twice and the base year will not reproduce.
* **The margin block.** National accounts offset trade and transport margins
  inside the commodity block: margin commodities sell services into every other
  commodity's supply column against a large NEGATIVE entry on their own
  diagonal. Ignoring it loses that entire value and shows up as a large residual
  on one commodity. The diagonal entries are pure accounting offsets appearing on
  both sides of the same balance, so dropping them preserves it exactly; what is
  left off-diagonal is economic, and enters as a Leontief requirement on top of
  the Armington composite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .sam import SAM
from .spec import AccountLayout, ModelSpec

__all__ = ["Economy", "calibrate"]


@dataclass
class Economy:
    """Everything the solver needs about the base year, in quantity units."""

    layout: AccountLayout
    spec: ModelSpec

    X0: np.ndarray            # activity output
    Q0: np.ndarray            # commodity output
    io: np.ndarray            # (n_com, n_act) intermediate per unit of activity output
    msh: np.ndarray           # (n_act, n_com) market shares, columns sum to one
    theta: np.ndarray         # (n_fac, n_act) CES value-added shares
    vunit: np.ndarray         # value added per unit of activity output
    aunit: np.ndarray         # (n_fac, n_act) factor per unit of output (Leontief branch)
    leontief: np.ndarray      # bool per activity
    own_account: np.ndarray   # bool: sells only to households, no commodity market share
    sigma_va: np.ndarray
    sigma_arm: np.ndarray

    ptax: np.ndarray          # product tax rate on absorption
    atax: np.ndarray          # specific tax per unit of activity output

    imp0: np.ndarray; exp0: np.ndarray; dom0: np.ndarray
    exp_share: np.ndarray
    dom_share: np.ndarray     # Armington domestic share
    A0: np.ndarray            # base absorption quantity, margins included
    mq: np.ndarray            # (n_com, n_com) margin per unit of absorption
    arm_share: np.ndarray     # Armington composite per unit of absorption

    hh_budget: np.ndarray     # (n_hh, n_com) commodity budget shares
    hh_savrate: np.ndarray
    hh_facshare: np.ndarray   # (n_hh, n_fac) ownership shares, columns sum to one
    hh_direct: np.ndarray     # (n_hh, n_act) payments made straight to an activity
    hh_committed: np.ndarray  # (n_hh, n_com) LES subsistence quantities
    hh_income0: np.ndarray
    pop_share: np.ndarray

    gov_dem: np.ndarray; inv_dem: np.ndarray
    fac_supply: np.ndarray
    trade_balance0: float
    foreign_sav: float

    diagnostics: dict = field(default_factory=dict, repr=False)

    #: CET elasticities by commodity. Used only when ``spec.export_rule == "cet"``.
    sigma_cet: np.ndarray | None = field(default=None, repr=False)

    # -- shorthands ------------------------------------------------------
    @property
    def acts(self) -> list: return self.layout.activities

    @property
    def coms(self) -> list: return self.layout.commodities

    @property
    def hhs(self) -> list: return self.layout.households

    @property
    def factors(self) -> list: return self.layout.factors

    @property
    def n_act(self) -> int: return len(self.acts)

    @property
    def n_com(self) -> int: return len(self.coms)

    @property
    def n_hh(self) -> int: return len(self.hhs)

    @property
    def n_fac(self) -> int: return len(self.factors)

    @property
    def numeraire_index(self) -> int:
        return self.factors.index(self.spec.numeraire_factor)

    def commodity(self, name: str) -> int: return self.coms.index(name)

    def activity(self, name: str) -> int: return self.acts.index(name)

    def household(self, name: str) -> int: return self.hhs.index(name)

    def factor(self, name: str) -> int: return self.factors.index(name)

    def __repr__(self) -> str:
        return (f"Economy({self.n_act} activities, {self.n_com} commodities, "
                f"{self.n_hh} households, {self.n_fac} factors)")


def calibrate(sam: SAM, spec: ModelSpec | None = None, *,
              income_elasticities: Mapping[str, float] | None = None,
              population: Mapping[str, float] | None = None,
              check_balance: bool = True) -> Economy:
    """Build an :class:`Economy` from a balanced SAM.

    Parameters
    ----------
    income_elasticities : mapping, optional
        Commodity -> income elasticity, used only when the household demand
        system is LES. Absent, every elasticity is one, which still gives a
        genuine LES because the subsistence quantities are nonzero.
    population : mapping, optional
        Household account -> number of households. This is the weight of a
        PER CAPITA transfer, and it is neither the income share nor the
        expenditure share. Getting it wrong reverses the distributional sign of
        any lump-sum recycling, so it is a separate input rather than a default
        derived from money.
    """
    spec = spec or ModelSpec()
    if check_balance:
        sam.check_balance()
    layout = spec.classify(sam)
    acts, coms, hhs, facs = (layout.activities, layout.commodities,
                             layout.households, layout.factors)
    na, nc, nh, nf = len(acts), len(coms), len(hhs), len(facs)
    gov, inv, row = layout.government, layout.investment, layout.rest_of_world

    receipts = sam.receipts()
    X0 = np.array([float(receipts.get(a, 0.0)) for a in acts])
    inter = sam.block(coms, acts)
    make = sam.block(acts, coms)
    fac = sam.block(facs, acts)

    Q0 = make.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        msh = np.where(Q0 > 0, make / np.maximum(Q0, 1e-12), 0.0)

    # An activity with no market share in any commodity does not appear in the
    # commodity balance, so its level cannot be recovered from `X = MSH @ Q` --
    # that would return zero and the base year would silently lose the whole
    # activity. Own-account production of this kind (self-generation, home
    # production, informal work) is carried at its base level and is the only
    # thing a scenario may scale exogenously.
    own_account = (np.abs(make).max(axis=1) < 1e-12) & (X0 > 0)

    # -- value added, with the negative-surplus treatment ------------------
    vunit = np.where(X0 > 0, fac.sum(axis=0) / np.maximum(X0, 1e-12), 0.0)
    aunit = np.where(X0 > 0, fac / np.maximum(X0, 1e-12)[None, :], 0.0)
    leontief = np.zeros(na, dtype=bool)
    theta = np.zeros_like(fac)
    negative = []
    for j in range(na):
        col = fac[:, j]
        if (col < 0).any():
            negative.append(acts[j])
            if spec.negative_value_added == "leontief":
                leontief[j] = True
            else:
                keep = np.where(col > 0, col, 0.0)
                net = col.sum()
                col = keep * (net / keep.sum()) if keep.sum() > 0 else keep
                aunit[:, j] = col / X0[j] if X0[j] > 0 else 0.0
        s = col.sum()
        theta[:, j] = col / s if s > 0 and not leontief[j] else 0.0
    if spec.value_added == "leontief":
        leontief[:] = True
        theta[:] = 0.0

    # Under "net" the sector's factor demand was removed, so aggregate supply
    # must fall by the same amount or the base year cannot clear.
    fac_supply = ((aunit * X0[None, :]).sum(axis=1)
                  if spec.negative_value_added == "net" else fac.sum(axis=1))

    # -- trade -------------------------------------------------------------
    imp0 = sam.block([row], coms).ravel()
    exp0 = sam.block(coms, [row]).ravel()
    exp_share = np.where(Q0 > 0, exp0 / np.maximum(Q0, 1e-12), 0.0)
    dom0 = Q0 - exp0

    # -- trade and transport margins ---------------------------------------
    CC = sam.block(coms, coms)
    np.fill_diagonal(CC, 0.0)
    if not spec.margins:
        CC = np.zeros_like(CC)
    margin_in = CC.sum(axis=0)

    # -- taxes -------------------------------------------------------------
    tax_c = sam.block([gov], coms).ravel()
    A0 = dom0 + imp0 + margin_in
    ptax = np.where(A0 > 0, tax_c / np.maximum(A0, 1e-12), 0.0)
    tax_a = sam.block([gov], acts).ravel()
    atax = np.where(X0 > 0, tax_a / np.maximum(X0, 1e-12), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        io = np.where(X0 > 0, inter / np.maximum(X0, 1e-12)[None, :], 0.0)
        io = io / (1.0 + ptax)[:, None]
        mq = (CC / (1.0 + ptax)[:, None]) / np.maximum(A0, 1e-12)[None, :]
        arm_share = (dom0 + imp0) / np.maximum(A0, 1e-12)

    # -- final demand ------------------------------------------------------
    hh_com = sam.block(coms, hhs).T                       # (n_hh, n_com), purchaser prices
    hh_budget = hh_com / np.maximum(hh_com.sum(axis=1, keepdims=True), 1e-12)
    # Household income is the account's FULL row total: factor income, transfers
    # from government or abroad, and anything else the SAM credits to it. Summing
    # a chosen list of columns instead silently drops whatever was not listed.
    hh_income0 = np.array([float(receipts.get(h, 0.0)) for h in hhs])
    hh_sav = sam.block([inv], hhs).ravel()
    hh_savrate = hh_sav / np.maximum(hh_income0, 1e-12)
    hh_fac = sam.block(hhs, facs)
    hh_facshare = hh_fac / np.maximum(hh_fac.sum(axis=0, keepdims=True), 1e-12)
    hh_direct = sam.block(acts, hhs).T                    # payments straight to an activity

    gov_dem = sam.block(coms, [gov]).ravel()
    inv_dem = sam.block(coms, [inv]).ravel()

    pop_share = _population_shares(hhs, population)
    hh_committed = _les_subsistence(spec, hh_com, hh_budget, ptax,
                                    income_elasticities, coms)

    trade_balance0 = float(imp0.sum() - exp0.sum())
    dom_share = np.where(dom0 + imp0 > 0, dom0 / np.maximum(dom0 + imp0, 1e-12), 1.0)

    econ = Economy(
        layout=layout, spec=spec, X0=X0, Q0=Q0, io=io, msh=msh, theta=theta,
        vunit=vunit, aunit=aunit, leontief=leontief, own_account=own_account,
        sigma_va=spec.sigma_va_vector(acts), sigma_arm=spec.sigma_armington_vector(coms),
        ptax=ptax, atax=atax, imp0=imp0, exp0=exp0, dom0=dom0, exp_share=exp_share,
        dom_share=dom_share, A0=A0, mq=mq, arm_share=arm_share,
        hh_budget=hh_budget, hh_savrate=hh_savrate, hh_facshare=hh_facshare,
        hh_direct=hh_direct, hh_committed=hh_committed, hh_income0=hh_income0,
        pop_share=pop_share, gov_dem=gov_dem, inv_dem=inv_dem, fac_supply=fac_supply,
        trade_balance0=trade_balance0, foreign_sav=trade_balance0,
        sigma_cet=spec.sigma_cet_vector(coms),
        diagnostics=dict(negative_value_added=negative,
                         negative_value_added_treatment=spec.negative_value_added,
                         n_leontief=int(leontief.sum()),
                         own_account=[acts[j] for j in np.where(own_account)[0]]),
    )
    return econ


def _population_shares(hhs, population) -> np.ndarray:
    if population is None:
        return np.full(len(hhs), 1.0 / len(hhs))
    missing = [h for h in hhs if h not in population]
    if missing:
        raise KeyError(f"no population given for household accounts {missing}")
    v = np.array([float(population[h]) for h in hhs])
    if np.any(v < 0):
        raise ValueError("negative population counts")
    return v / v.sum()


def _les_subsistence(spec, hh_com, hh_budget, ptax, income_elasticities, coms) -> np.ndarray:
    """Subsistence quantities implied by the Frisch parameter.

    Supernumerary income is ``Y* = -Y / frisch``, and each committed quantity
    follows from ``p_c gamma_c = p_c q_c - beta_c Y*`` with ``beta`` the marginal
    budget shares. Zero for Cobb-Douglas, which is the LES with no subsistence.
    """
    nh, nc = hh_com.shape
    if spec.household_demand != "les":
        return np.zeros((nh, nc))
    if spec.frisch >= 0:
        raise ValueError(f"the Frisch parameter must be negative, got {spec.frisch}")
    eta = np.ones(nc)
    if income_elasticities:
        for c, v in income_elasticities.items():
            if c in coms:
                eta[coms.index(c)] = float(v)
    beta = hh_budget * eta[None, :]
    beta = beta / np.maximum(beta.sum(axis=1, keepdims=True), 1e-12)
    Y = hh_com.sum(axis=1)
    Ystar = -Y / spec.frisch
    pq0 = 1.0 + ptax                                # base-year purchaser price
    value_gamma = hh_com - beta * Ystar[:, None]
    gamma = value_gamma / pq0[None, :]
    if np.any(gamma < -1e-9):
        raise ValueError(
            "the Frisch parameter implies a NEGATIVE subsistence quantity for at least one "
            "commodity, which is not a calibration but a contradiction: make `frisch` less "
            "negative (smaller supernumerary income) or supply income elasticities.")
    return np.maximum(gamma, 0.0)
