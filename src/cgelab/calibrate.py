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

from .mobility import build_mobility
from .vintages import VintageNest, build_vintages
from .segments import build_segments
from .sam import SAM
from .spec import AccountLayout, ModelSpec

__all__ = ["Economy", "calibrate"]


@dataclass
class EnergyNest:
    """Value-added-energy nesting, by activity. Values are per unit of activity output.

    ``vae_unit`` is value added plus energy at base-year prices, ``share`` the
    energy part of it, ``theta`` the fuel shares inside the energy bundle
    (energy rows only). Prices of both bundles are one at the base year.
    """

    mask: np.ndarray            # (n_com,) energy commodities
    vae_unit: np.ndarray        # (n_act,)
    share: np.ndarray           # (n_act,)
    theta: np.ndarray           # (n_com, n_act)
    sigma_vae: np.ndarray       # (n_act,), zero where fixed
    sigma_energy: np.ndarray    # (n_act,), zero where fixed
    #: ``"value_added"`` or ``"capital"``. Under ``"capital"``: the capital
    #: factor's index, the activities that substitute (``free``; the rest keep
    #: Leontief inputs), the top-nest shares of the other factors
    #: ``(n_fac, n_act)`` and of the capital-energy composite, capital's share
    #: inside that composite, and the two elasticities.
    mode: str = "value_added"
    capital: int = -1
    free: np.ndarray | None = None
    top_share: np.ndarray | None = None
    ke_share: np.ndarray | None = None
    k_share: np.ndarray | None = None
    sigma_ke: np.ndarray | None = None
    sigma_top: np.ndarray | None = None


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

    # -- institutions beyond the circular flow. Every one defaults to zero, so an
    # Economy built by hand, or from a SAM without these cells, behaves as before.
    #: (n_hh,) direct tax rate on gross income, from the SAM's gov <- hh cells.
    hh_dtax: np.ndarray | None = field(default=None, repr=False)
    #: (n_hh,) transfers from government, in base-year numeraire units (hh <- gov).
    hh_gov_transfer: np.ndarray | None = field(default=None, repr=False)
    #: (n_hh,) net transfers from abroad (hh <- row minus row <- hh), numeraire units.
    hh_remit_net: np.ndarray | None = field(default=None, repr=False)
    #: (n_fac,) shares of each factor's income paid to government and to the rest of
    #: the world. The household shares in `hh_facshare` apply to what is left.
    fac_gov_share: np.ndarray | None = field(default=None, repr=False)
    fac_row_share: np.ndarray | None = field(default=None, repr=False)
    #: (n_hh,) base-year spending on commodities -- the budget the LES was calibrated on.
    hh_spend0: np.ndarray | None = field(default=None, repr=False)
    #: (n_hh,) base-year gross income in the solver's sense: receipts net of
    #: transfers paid abroad. Equal to `hh_income0` when nothing is sent abroad.
    hh_gross0: np.ndarray | None = field(default=None, repr=False)

    #: Interregional nest. None when the model was not given ``regions``.
    regional: object | None = field(default=None, repr=False)

    #: Import tariff at the base year, Hosoe's ``taum = revenue / imports``.
    #: It is inside the import price the Armington nest was calibrated on, so an
    #: import is worth ``(1 + tm0)`` times its border value there. None means
    #: the instrument was not calibrated, and a scenario tariff is refused.
    tm0: np.ndarray | None = None
    #: Indirect tax on all of commodity output, Hosoe's ``tauz = revenue /
    #: output``. Home sales and exports both carry it: the CET splits output
    #: worth ``(1 + tauz0)`` times its producer value. None means the
    #: instrument was not calibrated.
    tauz0: np.ndarray | None = None
    #: Government saving over government income at the base year, income net
    #: of transfers to households. Read by ``Closure(government="saving_rate")``.
    gov_savrate: float = 0.0
    #: Government saving and total revenue (taxes and factor income, before
    #: transfers) at the base year: the default targets of
    #: ``Closure(government_target="saving")`` and ``"revenue"``.
    gov_saving0: float = 0.0
    gov_revenue0: float = 0.0
    #: Emissions account (:func:`cgelab.emissions.add_emissions`). None: no
    #: emissions are tracked and a carbon price is refused.
    emissions: object | None = field(default=None, repr=False)
    #: Physical units (:func:`cgelab.physical.add_physical_units`). None: money only.
    physical_units: object | None = field(default=None, repr=False)
    #: Value-added-energy nesting (``ModelSpec.energy``). None: Leontief inputs.
    energy_nest: EnergyNest | None = field(default=None, repr=False)
    #: Imperfectly mobile factors (``ModelSpec.factor_mobility``), by factor
    #: index: :class:`cgelab.mobility.FactorMobility`. Empty: every factor is
    #: perfectly mobile.
    mobility: dict = field(default_factory=dict, repr=False)
    #: Groups of labour segments (``ModelSpec.labour_segments``), by name:
    #: :class:`cgelab.segments.SegmentGroup`.
    segments: dict = field(default_factory=dict, repr=False)
    #: Putty-clay capital (``ModelSpec.vintages``): the two-bundle value added.
    #: None without vintages.
    vintage: VintageNest | None = field(default=None, repr=False)

    def zeros_if_none(self, name: str, n: int) -> np.ndarray:
        v = getattr(self, name)
        return np.zeros(n) if v is None else np.asarray(v, dtype=float)

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


def _tariff_and_output_tax(sam, spec, coms, gov, imp0, Q0, tariffs, output_taxes):
    """Base-year tariff and output-tax rates, or None when the instrument is absent."""
    nc = len(coms)
    if spec.tariff_account is not None and tariffs is not None:
        raise ValueError(
            "pass ModelSpec.tariff_account or calibrate(..., tariffs=), not both")
    if spec.indirect_tax_account is not None and output_taxes is not None:
        raise ValueError(
            "pass ModelSpec.indirect_tax_account or calibrate(..., output_taxes=), "
            "not both")
    tm0 = None
    if spec.tariff_account is not None:
        if spec.tariff_account not in sam.accounts:
            raise KeyError(f"tariff account {spec.tariff_account!r} is not in the SAM")
        revenue = sam.block([spec.tariff_account], coms).ravel()
        tm0 = _rates_from_revenue(revenue, imp0, coms, "tariff", "does not import")
    elif tariffs is not None:
        tm0 = _reference_rates(tariffs, nc, "tariffs")
        gov_com = sam.block([gov], coms).ravel()
        bad = [coms[i] for i in np.where(tm0 * imp0 > gov_com + 1e-8)[0]]
        if bad:
            raise ValueError(
                "tariff revenue implied by the rate exceeds what government already "
                f"receives from the commodity, for {bad}")
    tauz0 = None
    if spec.indirect_tax_account is not None:
        if spec.indirect_tax_account not in sam.accounts:
            raise KeyError(
                f"indirect tax account {spec.indirect_tax_account!r} is not in the SAM")
        revenue = sam.block([spec.indirect_tax_account], coms).ravel()
        tauz0 = _rates_from_revenue(revenue, Q0, coms, "indirect tax",
                                    "has no output to tax")
    elif output_taxes is not None:
        tauz0 = _reference_rates(output_taxes, nc, "output_taxes")
        gov_com = sam.block([gov], coms).ravel()
        bad = [coms[i] for i in np.where(tauz0 * Q0 > gov_com + 1e-8)[0]]
        if bad:
            raise ValueError(
                "output-tax revenue implied by the rate exceeds what government "
                f"already receives from the commodity, for {bad}")
    return tm0, tauz0


def _reference_rates(rates, n, label):
    out = np.asarray(rates, dtype=float)
    if out.shape != (n,):
        raise ValueError(f"{label} must have length {n}, got shape {out.shape}")
    if np.any(out <= -1.0):
        raise ValueError(f"a {label} rate must be greater than -1")
    return out


def _rates_from_revenue(revenue, base, names, label, empty):
    rates = np.zeros(len(names))
    for i, name in enumerate(names):
        if abs(revenue[i]) <= 1e-12:
            continue
        if base[i] <= 1e-12:
            raise ValueError(
                f"commodity {name!r} pays a {label} of {revenue[i]:.6g} but {empty}")
        rates[i] = revenue[i] / base[i]
    if np.any(rates <= -1.0):
        raise ValueError(f"a {label} rate must be greater than -1")
    return rates


def calibrate(sam: SAM, spec: ModelSpec | None = None, *,
              income_elasticities: Mapping[str, float] | None = None,
              population: Mapping[str, float] | None = None,
              tariffs=None,
              output_taxes=None,
              factor_pools: Mapping | None = None,
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
    tariffs, output_taxes : array, optional
        Reference rates, one per commodity, when the SAM has no separate
        tariff or indirect-tax account. Their revenue leaves the product tax
        and enters the import price and the gross output price, as in Hosoe's
        ``stdcge``. At those rates the base year is unchanged. A scenario then
        moves one instrument.
    factor_pools : mapping, optional
        Factor -> DataFrame of base-year use, pools of origin by activities:
        the proximity matrix of a factor in ``ModelSpec.factor_mobility``. Each
        column must add up to what the SAM pays the factor in that activity.
        Absent, the factor's whole endowment is one pool. ``"activities"``
        gives each activity that uses the factor a pool of its own, named
        after it: at any elasticity the factor is then sector-specific, and
        ``run_path`` accumulates a stock per activity.
    """
    spec = spec or ModelSpec()
    if check_balance:
        sam.check_balance()
    spec = spec.prepare(sam)
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

    # -- trade and transport margins ---------------------------------------
    CC = sam.block(coms, coms)
    np.fill_diagonal(CC, 0.0)
    if not spec.margins:
        CC = np.zeros_like(CC)
    margin_in = CC.sum(axis=0)

    # -- taxes -------------------------------------------------------------
    # Purchaser prices already contain every indirect tax buyers paid. The
    # reference rates below do not change that deflator: a scenario collects
    # the change from the calibrated rate, and the base year stays put.
    # A named account (TRF, IDT) is money buyers paid and government received,
    # recorded on its own row. Folding the receipt into the product tax keeps
    # 1 + ptax equal to the SAM's purchaser price. The account's payment back
    # to government is that same receipt, not a second one.
    tm0, tauz0 = _tariff_and_output_tax(
        sam, spec, coms, gov, imp0, Q0, tariffs, output_taxes)
    tax_c = sam.block([gov], coms).ravel()
    if spec.tariff_account is not None:
        tax_c = tax_c + sam.block([spec.tariff_account], coms).ravel()
    if spec.indirect_tax_account is not None:
        tax_c = tax_c + sam.block([spec.indirect_tax_account], coms).ravel()
    # The tariff and the output tax are inside the prices they fall on, as in
    # Hosoe's stdcge, and not in the product tax on absorption. An import enters
    # the Armington nest at its tariff-inclusive value, and output splits between
    # home sales and exports at its tax-inclusive value, so a domestic buyer
    # pays neither tax on the variety that did not carry it. Spreading them over
    # the whole composite instead would leave the domestic variety paying an
    # abolished tariff.
    tariff_rev0 = np.zeros(nc) if tm0 is None else tm0 * imp0
    output_rev0 = np.zeros(nc) if tauz0 is None else tauz0 * Q0
    tax_all_c = tax_c.copy()
    tax_c = tax_c - tariff_rev0 - output_rev0
    gross_out0 = Q0 + output_rev0                 # output at tax-inclusive value
    exp_share = np.where(gross_out0 > 0, exp0 / np.maximum(gross_out0, 1e-12), 0.0)
    dom0 = gross_out0 - exp0
    imp_arm0 = imp0 + tariff_rev0                 # imports at tariff-inclusive value
    if np.any(dom0 < -1e-9):
        bad = [coms[i] for i in np.where(dom0 < -1e-9)[0]]
        raise ValueError(f"exports exceed tax-inclusive output for {bad}")
    A0 = dom0 + imp_arm0 + margin_in
    ptax = np.where(A0 > 0, tax_c / np.maximum(A0, 1e-12), 0.0)
    tax_a = sam.block([gov], acts).ravel()
    atax = np.where(X0 > 0, tax_a / np.maximum(X0, 1e-12), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        io = np.where(X0 > 0, inter / np.maximum(X0, 1e-12)[None, :], 0.0)
        io = io / (1.0 + ptax)[:, None]
        mq = (CC / (1.0 + ptax)[:, None]) / np.maximum(A0, 1e-12)[None, :]
        arm_share = (dom0 + imp_arm0) / np.maximum(A0, 1e-12)

    # -- final demand ------------------------------------------------------
    hh_com = sam.block(coms, hhs).T                       # (n_hh, n_com), purchaser prices
    hh_budget = hh_com / np.maximum(hh_com.sum(axis=1, keepdims=True), 1e-12)
    # Household income is the account's FULL row total: factor income, transfers
    # from government or abroad, and anything else the SAM credits to it. Summing
    # a chosen list of columns instead silently drops whatever was not listed.
    hh_income0 = np.array([float(receipts.get(h, 0.0)) for h in hhs])
    hh_sav = sam.block([inv], hhs).ravel()
    for account in (spec.regional_investment or {}).values():
        hh_sav = hh_sav + sam.block([account], hhs).ravel()
    hh_tax = sam.block([gov], hhs).ravel()              # direct taxes
    gov_tr = sam.block(hhs, [gov]).ravel()              # transfers from government
    remit_in = sam.block(hhs, [row]).ravel()
    remit_out = sam.block([row], hhs).ravel()
    # Gross income in the model's sense: what the household disposes of before
    # tax and saving. Payments abroad are netted against receipts from abroad,
    # so they leave the budget without being taxed, saved or consumed.
    gross0 = hh_income0 - remit_out
    hh_dtax = np.where(gross0 > 0, hh_tax / np.maximum(gross0, 1e-12), 0.0)
    # Saving is a share of DISPOSABLE income. With no direct tax the two coincide
    # and the rate is the one every earlier version computed.
    hh_savrate = hh_sav / np.maximum(gross0 - hh_tax, 1e-12)
    hh_fac = sam.block(hhs, facs)
    hh_facshare = hh_fac / np.maximum(hh_fac.sum(axis=0, keepdims=True), 1e-12)
    # Factor income that does not reach resident households: to government
    # (public enterprises, land) and abroad (profits of foreign-owned capital,
    # cross-border wages). Dropping it would credit households with income they
    # never receive, and the base year would not reproduce.
    fac_gov = sam.block([gov], facs).ravel()
    fac_row = sam.block([row], facs).ravel()
    fac_tot = hh_fac.sum(axis=0) + fac_gov + fac_row
    fac_gov_share = np.where(fac_tot > 0, fac_gov / np.maximum(fac_tot, 1e-12), 0.0)
    fac_row_share = np.where(fac_tot > 0, fac_row / np.maximum(fac_tot, 1e-12), 0.0)
    hh_direct = sam.block(acts, hhs).T                    # payments straight to an activity

    gov_dem = sam.block(coms, [gov]).ravel()
    inv_dem = sam.block(coms, [inv]).ravel()

    pop_share = _population_shares(hhs, population)
    hh_committed = _les_subsistence(spec, hh_com, hh_budget, ptax,
                                    income_elasticities, coms)

    trade_balance0 = float(imp0.sum() - exp0.sum())
    # Foreign saving is the current-account deficit, not the trade deficit: factor
    # income paid abroad and net transfers abroad are part of it.
    foreign_sav0 = (trade_balance0 + float(fac_row.sum())
                    + float(remit_out.sum()) - float(remit_in.sum()))
    dom_share = np.where(dom0 + imp_arm0 > 0,
                         dom0 / np.maximum(dom0 + imp_arm0, 1e-12), 1.0)
    # Government income as the solver books it: every tax, its factor income,
    # net of its transfers to households. What it does not spend it saves.
    energy_nest = _energy_nest(spec, coms, acts, io, ptax, vunit, leontief, aunit, facs)
    vintage = build_vintages(spec, facs, acts, theta, leontief, spec.sigma_va_vector(acts))
    if vintage is not None:
        if vintage.old in (factor_pools or {}):
            raise ValueError(f"factor_pools names {vintage.old!r}, old capital, which "
                             f"vintages make sector-specific: a pool per activity")
        # Old capital is sector-specific: a pool per activity, which holds at
        # any elasticity. The economy records the spec with it, so every code
        # path that reads factor_mobility sees what was solved.
        import dataclasses
        spec = dataclasses.replace(
            spec, factor_mobility={**(spec.factor_mobility or {}), vintage.old: 0.0})
        factor_pools = {**dict(factor_pools or {}), vintage.old: "activities"}
    gov_income0 = (float(tax_all_c.sum()) + float(tax_a.sum()) + float(hh_tax.sum())
                   + float(fac_gov.sum()) - float(gov_tr.sum()))
    gov_savrate = (1.0 - float(gov_dem.sum()) / gov_income0) if gov_income0 > 0 else 0.0

    econ = Economy(
        layout=layout, spec=spec, X0=X0, Q0=Q0, io=io, msh=msh, theta=theta,
        vunit=vunit, aunit=aunit, leontief=leontief, own_account=own_account,
        sigma_va=spec.sigma_va_vector(acts), sigma_arm=spec.sigma_armington_vector(coms),
        ptax=ptax, atax=atax, imp0=imp0, exp0=exp0, dom0=dom0, exp_share=exp_share,
        dom_share=dom_share, A0=A0, mq=mq, arm_share=arm_share,
        hh_budget=hh_budget, hh_savrate=hh_savrate, hh_facshare=hh_facshare,
        hh_direct=hh_direct, hh_committed=hh_committed, hh_income0=hh_income0,
        pop_share=pop_share, gov_dem=gov_dem, inv_dem=inv_dem, fac_supply=fac_supply,
        trade_balance0=trade_balance0, foreign_sav=foreign_sav0,
        sigma_cet=spec.sigma_cet_vector(coms),
        hh_dtax=hh_dtax, hh_gov_transfer=gov_tr, hh_remit_net=remit_in - remit_out,
        fac_gov_share=fac_gov_share, fac_row_share=fac_row_share,
        hh_spend0=hh_com.sum(axis=1), hh_gross0=gross0,
        tm0=tm0, tauz0=tauz0, gov_savrate=gov_savrate,
        energy_nest=energy_nest, vintage=vintage,
        mobility=build_mobility(spec, facs, acts, aunit * X0[None, :], factor_pools),
        segments=build_segments(spec, facs, fac_supply),
        gov_saving0=gov_income0 - float(gov_dem.sum()),
        gov_revenue0=gov_income0 + float(gov_tr.sum()),
        diagnostics=dict(negative_value_added=negative,
                         negative_value_added_treatment=spec.negative_value_added,
                         n_leontief=int(leontief.sum()),
                         own_account=[acts[j] for j in np.where(own_account)[0]],
                         unmodelled_flows=_unmodelled_flows(
                             sam, layout,
                             [a for a in (spec.tariff_account,
                                          spec.indirect_tax_account) if a])),
    )
    if spec.regions:
        from .regional import build_regional
        econ.regional = build_regional(econ, sam, migration_pools=factor_pools)
    return econ


def _energy_nest(spec, coms, acts, io, ptax, vunit, leontief, aunit, facs):
    if not spec.energy:
        return None
    unknown = [c for c in spec.energy if c not in coms]
    if unknown:
        raise KeyError(f"energy commodities not in the model: {unknown}")
    mask = np.isin(np.asarray(coms, dtype=object), list(spec.energy))
    value = (1.0 + ptax)[:, None] * io                  # input value per unit output
    energy = value[mask].sum(axis=0)
    vae = vunit + energy
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(vae != 0, energy / np.where(vae == 0, 1.0, vae), 0.0)
        theta = np.where(energy[None, :] > 0,
                         value / np.where(energy > 0, energy, 1.0)[None, :], 0.0)
    theta[~mask] = 0.0
    # Substitution needs a positive value added; fixed proportions stay fixed.
    free = ~leontief & (vunit > 0)
    sig_vae = np.where(free, spec.sigma_vae_vector(acts), 0.0)
    sig_e = np.where(free, spec.sigma_energy_vector(acts), 0.0)
    if np.any(~np.isfinite(sig_vae)) or np.any(sig_vae < 0) or np.any(~np.isfinite(sig_e)) \
            or np.any(sig_e < 0):
        raise ValueError("sigma_vae and sigma_energy must be finite and non-negative")
    if spec.energy_nest == "value_added":
        return EnergyNest(mask=mask, vae_unit=vae, share=share, theta=theta,
                          sigma_vae=sig_vae, sigma_energy=sig_e)
    # Capital-energy composite inside the value-added nest. Factor payments per
    # unit of output are the base-year factor quantities (factor prices are 1).
    if spec.energy_capital not in facs:
        raise KeyError(f"energy_capital {spec.energy_capital!r} is not a factor ({list(facs)})")
    k = facs.index(spec.energy_capital)
    cap = aunit[k]
    free = free & (cap >= 0)
    ke = cap + energy
    with np.errstate(divide="ignore", invalid="ignore"):
        top = np.where(vae[None, :] > 0, aunit / np.where(vae > 0, vae, 1.0)[None, :], 0.0)
        ke_share = np.where(vae > 0, ke / np.where(vae > 0, vae, 1.0), 0.0)
        k_share = np.where(ke > 0, cap / np.where(ke > 0, ke, 1.0), 0.0)
    top[k] = 0.0
    top[:, ~free] = 0.0
    sig_top = (np.ones(len(acts)) if spec.value_added == "cobb_douglas"
               else spec.sigma_va_vector(acts))
    sig_ke = np.where(free, spec.sigma_ke_vector(acts), 0.0)
    if np.any(~np.isfinite(sig_ke)) or np.any(sig_ke < 0):
        raise ValueError("sigma_ke must be finite and non-negative")
    return EnergyNest(mask=mask, vae_unit=vae, share=share, theta=theta,
                      sigma_vae=sig_vae, sigma_energy=sig_e, mode="capital", capital=k,
                      free=free, top_share=top, ke_share=ke_share, k_share=k_share,
                      sigma_ke=sig_ke, sigma_top=np.where(free, sig_top, 0.0))


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


# Cells the model reads, as (row block, column block). A SAM entry outside this set
# is money the model never sees: it is not an error in itself (a stock-change
# account, a gov <- row grant), but the base year cannot reproduce with it and the
# reproduction test will say so without saying why. This list says why.
_READ_BLOCKS = {
    ("c", "a"), ("f", "a"), ("gov", "a"), ("a", "c"), ("row", "c"), ("gov", "c"),
    ("c", "c"), ("c", "hh"), ("c", "gov"), ("c", "inv"), ("c", "row"),
    ("hh", "f"), ("gov", "f"), ("row", "f"), ("a", "hh"),
    ("inv", "hh"), ("inv", "gov"), ("inv", "row"),
    ("gov", "hh"), ("hh", "gov"), ("hh", "row"), ("row", "hh"),
    ("tax", "c"), ("gov", "tax"),
}


def _unmodelled_flows(sam: SAM, layout: AccountLayout, tax_accounts=()) -> list:
    group = {}
    for a in layout.activities:
        group[a] = "a"
    for c in layout.commodities:
        group[c] = "c"
    for f in layout.factors:
        group[f] = "f"
    for h in layout.households:
        group[h] = "hh"
    group[layout.government] = "gov"
    group[layout.investment] = "inv"
    group[layout.rest_of_world] = "row"
    for account in tax_accounts:
        group[account] = "tax"
    out = []
    for r, c, v in sam.frame.itertuples(index=False):
        if abs(v) < 1e-12:
            continue
        if (group.get(r), group.get(c)) not in _READ_BLOCKS:
            out.append((r, c, float(v)))
    return out
