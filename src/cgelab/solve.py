"""Within-period general equilibrium.

Structure of one period
-----------------------
* **Production.** Leontief between the intermediate bundle and value added, and
  within the intermediate bundle; a configurable nest (CES, Cobb-Douglas or
  Leontief) over factors inside value added.
* **Make.** Commodity market shares are fixed at base-year proportions, so
  activity levels follow from commodity output, ``X = MSH @ Q``.
* **Armington.** Domestic and imported varieties combine by CES.
* **Households.** Cobb-Douglas or LES over commodities, on income net of saving
  and of any payment made straight to an activity.
* **Closure.** See :class:`cgelab.closure.Closure`.

Why the quantity system is written on COMMODITY output
------------------------------------------------------
With more commodities than activities the commodity balances cannot determine
activity levels directly: that system is overdetermined, and solving it by least
squares would be arithmetic dressed as economics. So the balances determine
commodity output and activity levels follow from the market shares::

    (I - B G - MQ) A = final,   G = diag(dsh * arm_share / (1 - export share)),
    B = IO @ MSH,               Q = G A,   X = MSH @ Q

What is deliberately NOT solved
-------------------------------
The numeraire factor's market. It is left to Walras' law so that its residual is
an independent check on the accounting rather than an identity the solver was
told to satisfy, and it is recorded on every period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from .calibrate import Economy
from .closure import Closure
from .functions import (armington, ces_intensity, ces_price, cet, cet_domestic_price,
                        cobb_douglas_price)
from .mobility import period_endowments
from .segments import period_groups
from .scenario import Scenario, investment_shares

__all__ = ["PeriodResult", "solve_period", "value_added_price", "factor_intensity"]

_PRICE_ITERS = 200
_RECYCLE_TOL = 1e-13


# ---------------------------------------------------------------------------
# Price blocks
# ---------------------------------------------------------------------------

def _factor_price_matrix(econ: Economy, w: np.ndarray) -> np.ndarray:
    """Factor prices as ``(n_fac, n_act)``: a vector is the same price everywhere."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        return np.broadcast_to(w[:, None], (w.size, econ.n_act))
    if w.shape != (econ.n_fac, econ.n_act):
        raise ValueError(f"factor prices must be ({econ.n_fac},) or "
                         f"({econ.n_fac}, {econ.n_act}), got {w.shape}")
    return w


def value_added_price(econ: Economy, w: np.ndarray) -> np.ndarray:
    """Unit cost index of value added, by activity.

    ``w`` is one price per factor, or ``(n_fac, n_act)`` when a factor earns a
    different price in each activity (``ModelSpec.factor_mobility``).
    """
    w = _factor_price_matrix(econ, w)
    if econ.spec.value_added == "cobb_douglas":
        nest = cobb_douglas_price(econ.theta, w)
    else:
        nest = ces_price(econ.theta, w, econ.sigma_va)
    leo = (w * econ.aunit).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        leo_unit = np.where(econ.vunit != 0, leo / np.where(econ.vunit == 0, 1.0, econ.vunit), 0.0)
    return np.where(econ.leontief, leo_unit, nest)


def factor_intensity(econ: Economy, w: np.ndarray, *, vunit=None, aunit=None) -> np.ndarray:
    """Factor input per unit of value added, ``(n_fac, n_act)``.

    ``w`` is a vector or a matrix, as in :func:`value_added_price`. ``vunit``
    and ``aunit`` override the calibrated coefficients, which is how a
    productivity path enters: the CES branch is scale-free and unaffected, but
    the fixed-proportions branch is a ratio of the two and must see both.
    """
    vunit = econ.vunit if vunit is None else vunit
    aunit = econ.aunit if aunit is None else aunit
    w = _factor_price_matrix(econ, w)
    pva = value_added_price(econ, w)
    if econ.spec.value_added == "cobb_douglas":
        with np.errstate(divide="ignore", invalid="ignore"):
            nest = np.where(econ.theta > 0, econ.theta * pva[None, :] / w, 0.0)
    else:
        nest = ces_intensity(econ.theta, w, pva, econ.sigma_va)
    with np.errstate(divide="ignore", invalid="ignore"):
        leo = np.where(vunit != 0, aunit / np.where(vunit == 0, 1.0, vunit), 0.0)
    return np.where(econ.leontief[None, :], leo, nest)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PeriodResult:
    """Solved equilibrium for one period, with its own convergence evidence."""

    econ: Economy = field(repr=False)
    closure: Closure = field(repr=False)
    scenario: Scenario = field(repr=False)
    z: np.ndarray
    factor_prices: np.ndarray
    exchange_rate: float
    pa: np.ndarray; pd: np.ndarray; pq: np.ndarray; pqf: np.ndarray; pbase: np.ndarray
    Q: np.ndarray; X: np.ndarray; absorption: np.ndarray
    imports: np.ndarray; exports: np.ndarray
    hh_quantity: np.ndarray; hh_income: np.ndarray; hh_commodity_spend: np.ndarray
    hh_direct_outlay: np.ndarray; hh_wedge_paid: np.ndarray
    gov_quantity: np.ndarray; investment_quantity: np.ndarray
    factor_demand: np.ndarray
    #: Factor use by activity, ``(n_fac, n_act)``, with the scenario's productivity
    #: already in the coefficients. The row sums are ``factor_demand``. Kept on the
    #: result so a user does not rebuild the productivity scaling by hand and get
    #: a different employment number from the one the equilibrium was solved at.
    factor_demand_by_activity: np.ndarray
    savings: float; wedge_revenue: float
    government_cost: float
    real: dict
    gaps: dict
    walras: float
    residual: float
    converged: bool
    attempt: str = ""
    message: str = ""
    pe: np.ndarray | None = None      # price received for exports
    hh_direct_tax: np.ndarray | None = None     # direct tax paid, by household
    factor_income_abroad: float = 0.0           # factor income paid to the rest of the world
    foreign_saving: float = 0.0                 # the period's foreign saving (FDI included)
    #: Labour factor. ``employment`` is demand; ``unemployment`` is the labour
    #: force minus that, and is zero when the market clears or supply is elastic.
    #: ``real_wage`` is the nominal wage over the base-weighted purchaser index.
    employment: float = float("nan")
    unemployment: float = 0.0
    real_wage: float = float("nan")
    #: Domestic origin sold into each destination's absorption,
    #: ``(origin, destination, local commodity)``. Government and investment are
    #: national and are not a destination. None when the model has no regions.
    bilateral: np.ndarray | None = None
    #: World prices in foreign currency, as multiples of the base year: the
    #: scenario's ``export_price`` and ``world_price``, moved along the demand
    #: and supply curves of ``ModelSpec.export_demand`` and ``import_supply``.
    world_export_price: np.ndarray | None = None
    world_import_price: np.ndarray | None = None
    #: Home sales of domestic output: the domestic variety in the Armington
    #: nest, at its tax-inclusive base value, and the price it fetches. Off the
    #: base year a CET does not add up in quantities, so this is not output
    #: less exports.
    domestic_sales: np.ndarray | None = None
    p_domestic: np.ndarray | None = None
    #: The government account, in current prices: revenue by instrument
    #: (``product_tax``, ``activity_tax``, ``carbon_production``,
    #: ``carbon_households``, ``tariff``, ``output_tax``,
    #: ``direct_tax``, ``factor_income``, ``wedge``), their sum ``revenue``,
    #: outlays (``transfers``, ``consumption``, ``absorbed``) and ``saving``.
    government: dict | None = None
    #: Voluntary household saving, and the forced saving (a per-capita levy)
    #: that finances fixed real investment. Zero under savings-driven investment.
    household_saving: float = 0.0
    forced_saving: float = 0.0
    #: The common factor on the adjusting tax's rates under a government
    #: target (``Closure.government_adjusts``); one otherwise.
    tax_scale: float = 1.0
    #: Price of each factor in each activity, ``(n_fac, n_act)``. A perfectly
    #: mobile factor earns ``factor_prices`` everywhere. An imperfectly mobile
    #: one (``ModelSpec.factor_mobility``) earns its own price in each activity
    #: that uses it; for it, ``factor_prices`` is its income over its endowment
    #: and ``factor_demand`` is the endowment, so their product is still the
    #: factor's income. Under a CET, the row of ``factor_demand_by_activity``
    #: does not add up to the endowment off the base year.
    factor_prices_by_activity: np.ndarray | None = None
    #: Pools of each imperfectly mobile factor, ``{factor: {"pools", "price",
    #: "endowment", "supply"}}``, ``supply`` being ``(n_pool, n_act)``. None
    #: when every factor is perfectly mobile.
    factor_pools: dict | None = None
    #: Each group of labour segments (``ModelSpec.labour_segments``):
    #: ``{group: {"segments", "informal", "wage", "real_wage", "employment",
    #: "labour_force", "unemployment", "unemployment_rate", "harris_todaro",
    #: "household_workforce", "household_employment", "household_unemployed"}}``,
    #: household employment being ``(n_hh, n_seg)``. None without segments.
    labour_markets: dict | None = None
    #: Intermediate input per unit of activity output, ``(n_com, n_act)``, and
    #: primary factor use per unit (in base-year value), as solved: with an
    #: energy nest both answer to prices, and these are the coefficients the
    #: equilibrium used.
    input_coefficients: np.ndarray | None = None
    value_added_per_unit: np.ndarray | None = None

    def price(self, commodity: str) -> float:
        return float(self.pq[self.econ.commodity(commodity)])

    def household_real_income(self) -> np.ndarray:
        return self.real["hh_real_income"]

    def summary(self) -> str:
        e = self.econ
        lines = [f"equilibrium ({self.scenario.name}): {self.closure.describe()}",
                 f"  residual        : {self.residual:.3e} "
                 f"({'converged' if self.converged else 'NOT CONVERGED'}, {self.attempt})",
                 f"  Walras residual : {self.walras:.3e}  "
                 f"[{e.spec.numeraire_factor}, not solved for]"]
        for f, p in zip(e.factors, self.factor_prices):
            lines.append(f"  price {f:<12}: {p:.6f}")
        lines.append(f"  exchange rate   : {self.exchange_rate:.6f}")
        lines.append(f"  savings         : {self.savings:,.1f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_period(econ: Economy, closure: Closure | None = None,
                 scenario: Scenario | None = None, *, warm_start=None,
                 raise_on_fail: bool = True) -> PeriodResult:
    """Solve one period's general equilibrium.

    Unknowns
    --------
    One price per non-numeraire factor, plus one unknown for the external
    closure -- the exchange rate under ``foreign="fixed_saving"``, foreign saving
    itself under ``"fixed_rate"`` -- plus, under ``investment="fixed_real"``, the
    forced household saving that finances the fixed investment. A fixed real wage
    drops the labour price: the wage is the price index times the target, and
    employment is whatever labour demand that wage produces. Every closure
    therefore carries its own adjusting variable; a closure that pins a magnitude
    without freeing one somewhere else does not close the model, and the symptom
    is a Walras residual rather than a solver failure.
    """
    closure = closure or Closure()
    sc = scenario or Scenario()
    if econ.regional is not None:
        from .regional import solve_regions
        return solve_regions(econ, closure, sc, warm_start=warm_start,
                             raise_on_fail=raise_on_fail)
    nf, nc, nh, na = econ.n_fac, econ.n_com, econ.n_hh, econ.n_act
    num = econ.numeraire_index
    lab = (econ.factor(closure.labour_factor)
           if closure.labour_factor in econ.factors else None)
    mob = econ.mobility
    if closure.labour != "full_employment" and lab in mob:
        raise ValueError(
            f"closure.labour={closure.labour!r} on {closure.labour_factor!r}, which is not "
            f"perfectly mobile (ModelSpec.factor_mobility). With a wage in every activity "
            f"there is no single wage for the closure to set; only 'full_employment' is "
            f"available for it.")
    if closure.labour != "full_employment":
        if lab is None:
            raise ValueError(
                f"closure.labour_factor {closure.labour_factor!r} is not a factor "
                f"of this economy ({list(econ.factors)})")
        if lab == num:
            raise ValueError(
                f"closure.labour={closure.labour!r} cannot use {closure.labour_factor!r} "
                f"as the numeraire. That market is left to Walras' law, so pinning or "
                f"linking its wage would turn the check into an identity or a leak. "
                f"Set ModelSpec.numeraire_factor to another factor.")
    supply, endow = period_endowments(econ, sc)
    base_use = econ.aunit * econ.X0[None, :]
    facshare = (econ.hh_facshare if sc.factor_ownership is None
                else np.asarray(sc.factor_ownership, float))
    groups = period_groups(econ, closure, sc, supply, facshare)
    if closure.labour != "full_employment" and any(
            lab in g.group.index for g in groups):
        raise ValueError(
            f"closure.labour={closure.labour!r} on {closure.labour_factor!r}, which is a "
            f"labour segment. Its wage is set by Closure.labour_markets; make it rigid there.")
    # A segment that is rigid, or that shares the informal wage, has no price of
    # its own to solve for. Its employment is what demand asks for.
    tied = {i for g in groups for i in g.hired}
    # A household's labour income from a group follows its workers, not a fixed
    # share of each segment, so the group's columns leave the ownership matrix.
    facshare_eff = np.array(facshare, dtype=float)
    for g in groups:
        facshare_eff[:, g.group.index] = 0.0

    # A fixed real wage is not an unknown: it is the price index times the target.
    # Dropping it without dropping the market equation would overdetermine the model.
    # An imperfectly mobile factor has no single price: its unknowns are its
    # prices in each activity that uses it, appended after every other block.
    others = [i for i in range(nf) if i != num and i not in mob and i not in tied
              and not (closure.labour == "fixed_real_wage" and i == lab)]
    i_ext = len(others)                       # external-closure unknown
    i_inv = i_ext + 1                         # forced-saving unknown, when present
    n_unknown = i_ext + 1 + (1 if closure.investment == "fixed_real" else 0)
    tfp = np.ones(na) if sc.productivity is None else np.asarray(sc.productivity, float)
    ptax_a_set = econ.ptax if sc.product_tax is None else np.asarray(sc.product_tax, float)
    ptax_f_set = (ptax_a_set if sc.product_tax_firm is None
                  else np.asarray(sc.product_tax_firm, float))
    world_p = np.ones(nc) if sc.world_price is None else np.asarray(sc.world_price, float)
    tm0 = econ.tm0
    if sc.tariff is not None and tm0 is None:
        raise ValueError(
            "Scenario.tariff is read only when a base-year tariff was calibrated, "
            "via calibrate(..., tariffs=) or ModelSpec.tariff_account. Without that, "
            "the rate would be invented.")
    tm = tm0 if sc.tariff is None else np.asarray(sc.tariff, dtype=float)
    if tm is not None and tm.shape != (nc,):
        raise ValueError(f"tariff must have length {nc}, got shape {tm.shape}")
    tauz0 = econ.tauz0
    if sc.output_tax is not None and tauz0 is None:
        raise ValueError(
            "Scenario.output_tax is read only when an indirect tax on output was "
            "calibrated, via calibrate(..., output_taxes=) or "
            "ModelSpec.indirect_tax_account.")
    tauz = tauz0 if sc.output_tax is None else np.asarray(sc.output_tax, dtype=float)
    # Tax-inclusive base values over producer or border values. One where the
    # instrument was not calibrated.
    tf0 = np.ones(nc) if tm0 is None else 1.0 + tm0
    tz0 = np.ones(nc) if tauz0 is None else 1.0 + tauz0
    if tauz is not None and tauz.shape != (nc,):
        raise ValueError(f"output_tax must have length {nc}, got shape {tauz.shape}")
    cet_on = econ.spec.export_rule == "cet"
    if sc.export_price is not None and not cet_on:
        raise ValueError(
            "Scenario.export_price is read only under export_rule='cet'. With fixed export "
            "shares exports are a fixed fraction of output whatever they fetch abroad, so a "
            "world export price would be silently ignored.")
    world_pe = np.ones(nc) if sc.export_price is None else np.asarray(sc.export_price, float)
    sigma_cet = (econ.sigma_cet if econ.sigma_cet is not None
                 else econ.spec.sigma_cet_vector(econ.coms))
    # Large-country trade. A commodity on a curve has its world price as an
    # outer unknown, in logs relative to the scenario's shift, and the curve is
    # its residual. A commodity the base year neither exports nor imports has no
    # base volume to anchor a curve on and stays a price taker.
    eta_x, psi_m, curve_x, curve_m = _trade_curves(econ)
    imported = econ.imp0 > 0
    if (not cet_on and closure.solves_exchange_rate
            and closure.foreign_saving_unit == "numeraire" and curve_m.size == 0
            and imported.any() and np.all(np.abs(econ.sigma_arm[imported] - 1.0) < 1e-9)):
        # Imports are then a fixed value share of absorption and exports a fixed
        # value share of output, so no relative price moves the current account.
        # Any exchange rate clears it, with the same output and factor prices; the
        # solver would return whichever its start point happened to be.
        raise ValueError(
            "the exchange rate is not determined: with export_rule='fixed_shares', a "
            "Cobb-Douglas Armington (sigma_armington = 1 on every imported commodity) and "
            "foreign saving fixed in numeraire units, every exchange rate is an "
            "equilibrium. Use export_rule='cet', an Armington elasticity other than 1, "
            "Closure(foreign_saving_unit='foreign_currency') or foreign='fixed_rate'. "
            "Close to 1 the model is determined but ill-conditioned: the exchange rate "
            "swings as 1 / (sigma - 1).")
    i_curve = n_unknown
    n_unknown += curve_x.size + curve_m.size
    # A government target met by a tax is one more outer unknown: the log of a
    # common factor on that tax's rates. The product tax enters the price fixed
    # point, so it cannot be solved inside the affine budget block.
    target = closure.government_target
    adjusts = closure.government_adjusts
    tax_lever = target is not None and adjusts in ("direct_tax", "product_tax")
    i_gov = n_unknown
    n_unknown += 1 if tax_lever else 0
    i_mob = {}
    for i, m in mob.items():
        i_mob[i] = np.arange(n_unknown, n_unknown + int(m.users.sum()))
        n_unknown += int(m.users.sum())
    # Harris–Todaro unemployment, as its departure from the base year's in units
    # of the base year's rigid employment. Not in logs: with a small base rate
    # the residual hardly moves with log U, and Newton's first step overflows.
    i_ht = {}
    for k, g in enumerate(groups):
        if g.harris_todaro:
            i_ht[k] = n_unknown
            n_unknown += 1
    # Every wage pinned to the price index: the labour closure's and each rigid
    # segment's.
    rigid_wages = ([(lab, 1.0 if closure.real_wage is None else float(closure.real_wage))]
                   if closure.labour == "fixed_real_wage" else [])
    for g in groups:
        rigid_wages += list(zip(g.rigid, g.target))
    # Employment that is labour demand, solved in closed form below.
    slack_idx = (([lab] if closure.labour in ("fixed_real_wage", "wage_curve") else [])
                 + [i for g in groups for i in g.hired])
    if target == "saving":
        target_value = (econ.gov_saving0 if closure.government_target_value is None
                        else float(closure.government_target_value)) * closure.numeraire
    elif target == "revenue":
        target_value = (econ.gov_revenue0 if closure.government_target_value is None
                        else float(closure.government_target_value)) * closure.numeraire
    else:
        target_value = 0.0
    W = np.zeros((nh, nc)) if sc.consumer_wedge is None else np.asarray(sc.consumer_wedge, float)
    Wg_set = (np.zeros((nh, nc)) if sc.government_wedge is None
              else np.asarray(sc.government_wedge, float))
    base_transfers = np.zeros(nh) if sc.transfers is None else np.asarray(sc.transfers, float)
    dtax_set = econ.zeros_if_none("hh_dtax", nh)
    if tax_lever and adjusts == "direct_tax" and not np.any(dtax_set != 0):
        raise ValueError("government_adjusts='direct_tax', but no household pays direct tax; "
                         "a zero rate scaled by any factor stays zero")
    if tax_lever and adjusts == "product_tax" and not np.any(ptax_a_set != 0):
        raise ValueError("government_adjusts='product_tax', but every product tax rate is zero")
    fac_gov_sh = econ.zeros_if_none("fac_gov_share", nf)
    fac_row_sh = (econ.zeros_if_none("fac_row_share", nf) if sc.foreign_factor_share is None
                  else np.asarray(sc.foreign_factor_share, float))
    if np.any(fac_gov_sh + fac_row_sh > 1.0 + 1e-12) or np.any(fac_row_sh < 0):
        raise ValueError("factor shares paid to government and abroad must lie in [0, 1] "
                         "and sum to at most one for every factor")
    for f in sc.fdi:
        econ.factor(f)                    # a KeyError naming the factor, not a silent no-op

    # Own-account activities are exogenous by construction, at their base level
    # unless a scenario scales them: they are absent from the commodity balance,
    # so `X = MSH @ Q` cannot recover them and would silently return zero.
    act_scale = np.ones(na)
    exog = econ.own_account.copy()
    for name, mult in sc.activity_scale.items():
        j = econ.activity(name) if isinstance(name, str) else name
        if not econ.own_account[j]:
            raise ValueError(
                f"activity {econ.acts[j]!r} has a market share in the commodity block, so its "
                f"level is determined by the commodity balance and cannot also be set "
                f"exogenously. `activity_scale` is for own-account production sold straight "
                f"to households.")
        act_scale[j] = float(mult)

    # Nominal magnitudes denominated in the numeraire co-scale with it.
    nu = closure.numeraire
    # The current-account target IS foreign saving: one magnitude, read from one
    # field. `trade_balance0` stays on the Economy as the base-year record, but a
    # scenario that moved `foreign_sav` alone used to leave the target at its old
    # value, and the gap surfaced as a Walras residual.
    # FDI is foreign saving that arrives this period, so it is part of the target.
    # In foreign currency the target is not co-scaled with the numeraire: the
    # exchange rate does that, because it is what turns dollars into the
    # numeraire's units.
    in_dollars = closure.foreign_saving_unit == "foreign_currency"
    trade_target = (econ.foreign_sav + float(sum(sc.fdi.values()))) * (1.0 if in_dollars else nu)
    gov_tr = econ.zeros_if_none("hh_gov_transfer", nh) * nu     # fixed in numeraire units,
    remit = econ.zeros_if_none("hh_remit_net", nh) * nu         # like foreign saving
    atax_set = econ.atax * nu
    # A carbon price is a specific tax: on an activity, per unit of output times
    # its emission intensity at the scenario's input coefficients; on
    # households, per unit of fuel bought. Both in numeraire units.
    if sc.carbon_price is not None and econ.emissions is None:
        raise ValueError("Scenario.carbon_price needs an emissions account: see add_emissions")
    carbon = 0.0 if sc.carbon_price is None else float(sc.carbon_price) * nu
    ext_scale = max(abs(trade_target), 1.0)
    sav_scale = max(abs(float(econ.inv_dem.sum())) * nu, 1.0)

    # Hicks-neutral productivity scales EVERY input requirement, so it enters the
    # coefficient matrices and not only the price block. Applying it to the unit
    # cost alone leaves factor and intermediate demand at the old intensities:
    # the price system then says one thing and the quantity system another, and
    # the difference surfaces as a Walras residual that GROWS with the
    # productivity path -- invisible in a single period, unmistakable over a
    # twenty-five year horizon.
    io_eff = econ.io / tfp[None, :]
    nest = econ.energy_nest
    emask = nest.mask if nest is not None else np.zeros(nc, dtype=bool)
    # Under an energy nest, fuel use answers to prices and the carbon on fuel
    # enters the fuel price each activity pays. What stays a tax per unit of
    # output is process carbon and carbon on non-energy inputs.
    if carbon:
        ef_fixed = np.where(emask[:, None], 0.0, econ.emissions.intermediate)
        carb_act = carbon * ((ef_fixed * io_eff).sum(axis=0) + econ.emissions.process)
        carb_fuel = carbon * np.where(emask[:, None], econ.emissions.intermediate, 0.0)
    else:
        carb_act, carb_fuel = np.zeros(na), np.zeros((nc, na))
    carb_hh = carbon * econ.emissions.household if carbon else np.zeros(nc)
    atax = atax_set + carb_act
    vunit_eff = econ.vunit / tfp
    aunit_eff = econ.aunit / tfp[None, :]
    io_fixed = np.where(emask[:, None], 0.0, io_eff)    # Leontief inputs
    p_fuel0 = (1.0 + econ.ptax)[:, None]               # base-year fuel prices

    def production(pqf, w, quantities=True, pva=None):
        """Unit cost by activity, input coefficients and factor use per unit of output.

        ``w`` is the factor price matrix, ``(n_fac, n_act)``.

        Without an energy nest these are the Leontief coefficients and the
        value-added nest. With one, the energy bundle (a CES across fuels, at
        the fuel prices each activity pays, carbon included) either substitutes
        for value added as a whole in a VAE bundle (``"value_added"``), or joins
        a capital factor in a capital-energy composite that sits inside the
        value-added nest (``"capital"``). Every bundle is in base-year value and
        its price is one at the base year.
        """
        pva = value_added_price(econ, w) if pva is None else pva
        if nest is None:
            cost = io_eff.T @ pqf + pva * vunit_eff + atax
            if not quantities:          # the price fixed point needs the cost only
                return cost, None, None
            intensity = factor_intensity(econ, w, vunit=vunit_eff, aunit=aunit_eff)
            return cost, io_eff, intensity * vunit_eff[None, :]
        fuel_rel = (pqf[:, None] + carb_fuel) / p_fuel0            # (nc, na)
        sig_e, theta = nest.sigma_energy, nest.theta
        vae_q = nest.vae_unit / tfp
        with np.errstate(divide="ignore", invalid="ignore"):
            has_e = theta.sum(axis=0) > 0
            pe = np.where(has_e, _energy_index(theta, fuel_rel, sig_e), 1.0)
            if nest.mode == "value_added":
                sig_v, sh = nest.sigma_vae, nest.share
                pvae = _two_input_ces(1.0 - sh, pva, sh, pe, sig_v)
                va_per = vae_q * (1.0 - sh) * _ratio_power(pvae, pva, sig_v)
                e_per = vae_q * sh * _ratio_power(pvae, pe, sig_v)
                top_cost = vae_q * pvae
                if not quantities:
                    return io_fixed.T @ pqf + top_cost + atax, None, None
                fac_per = factor_intensity(econ, w, vunit=vunit_eff,
                                           aunit=aunit_eff) * va_per[None, :]
            else:
                k, free = nest.capital, nest.free
                sk, s_ke, s_top = nest.k_share, nest.sigma_ke, nest.sigma_top
                pke = _two_input_ces(sk, w[k], 1.0 - sk, pe, s_ke)
                prices = np.vstack([w, pke[None, :]])
                shares = np.vstack([nest.top_share, nest.ke_share[None, :]])
                ptop = _energy_index(shares, prices, s_top)
                ke_per = vae_q * nest.ke_share * _ratio_power(ptop, pke, s_top)
                e_per = ke_per * (1.0 - sk) * _ratio_power(pke, pe, s_ke)
                fac_per = (vae_q[None, :] * nest.top_share
                           * _ratio_power(ptop[None, :], w, s_top[None, :]))
                fac_per[k] = ke_per * sk * _ratio_power(pke, w[k], s_ke)
                top_cost = vae_q * ptop
                # Activities without substitution keep their Leontief inputs.
                fac_per[:, ~free] = aunit_eff[:, ~free]
                e_per = np.where(free, e_per, 0.0)
                top_cost = np.where(free, top_cost, (w * aunit_eff).sum(axis=0)
                                    + ((pqf[:, None] + carb_fuel) * np.where(
                                        emask[:, None], io_eff, 0.0)).sum(axis=0))
            fuel = np.where(emask[:, None], e_per[None, :] * theta
                            * _ratio_power(pe[None, :], fuel_rel, sig_e[None, :]) / p_fuel0,
                            0.0)
        if nest.mode == "capital":
            fuel[:, ~nest.free] = np.where(emask[:, None], io_eff, 0.0)[:, ~nest.free]
        coef = io_fixed + fuel
        cost = io_fixed.T @ pqf + top_cost + atax
        return cost, coef, fac_per

    X_exog = econ.X0 * act_scale * exog

    alpha_recycle = _recycle_shares(econ, sc, facshare, supply)
    beta_les = _marginal_budget_shares(econ)
    inv_share = investment_shares(econ, sc)
    inv_budget0 = float(econ.inv_dem.sum())
    state: dict = {}

    gov_w = econ.gov_dem / max(float(econ.gov_dem.sum()), 1e-300)
    gov_savrate = float(econ.gov_savrate)

    def domestic_price(p_gross, pe):
        """Price of domestic sales: the tax-inclusive output price, or its CET split."""
        return cet_domestic_price(econ.exp_share, p_gross, pe, sigma_cet) if cet_on else p_gross

    def residuals(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        w = np.full(nf, nu)
        for k, i in enumerate(others):
            w[i] = nu * float(np.exp(z[k]))
        for g in groups:
            w[g.flexible] = w[g.informal]
        U_now = [g.U0 + float(z[i_ht[k]]) * g.rigid_base if k in i_ht else 0.0
                 for k, g in enumerate(groups)]
        Wm = np.tile(w[:, None], (1, na))
        for i, m in mob.items():
            Wm[i, m.users] = nu * np.exp(z[i_mob[i]])
        if closure.solves_exchange_rate:
            er = float(np.exp(z[i_ext]))
            foreign_sav = er * trade_target if in_dollars else trade_target
        else:
            er = closure.exchange_rate
            foreign_sav = trade_target + float(z[i_ext]) * ext_scale
        levy = float(z[i_inv]) * sav_scale if closure.investment == "fixed_real" else 0.0
        tax_scale = float(np.exp(z[i_gov])) if tax_lever else 1.0
        ptax_a = ptax_a_set * (tax_scale if adjusts == "product_tax" and tax_lever else 1.0)
        ptax_f = ptax_f_set * (tax_scale if adjusts == "product_tax" and tax_lever else 1.0)
        dtax = dtax_set * (tax_scale if adjusts == "direct_tax" and tax_lever else 1.0)
        log_px = z[i_curve:i_curve + curve_x.size]
        log_pm = z[i_curve + curve_x.size:i_curve + curve_x.size + curve_m.size]
        world_p_now = world_p.copy()
        world_p_now[curve_m] *= np.exp(log_pm)
        world_pe_now = world_pe.copy()
        world_pe_now[curve_x] *= np.exp(log_px)
        # Border price is what the balance of payments pays. The Armington import
        # is measured at its tariff-inclusive base value, so its price is the
        # border price times (1 + tm) / (1 + tm0): one at the base year, as the
        # rest of the base year is. Output is measured the same way at its
        # tax-inclusive value, and its gross price is the producer price times
        # (1 + tauz) / (1 + tauz0).
        pm_border = er * world_p_now
        pm = pm_border if tm is None else pm_border * (1.0 + tm) / tf0
        out_wedge = 1.0 if tauz is None else (1.0 + tauz) / tz0
        pe = er * world_pe_now

        # --- prices: a fixed point, because margins are themselves delivered goods.
        # A fixed real wage joins the same loop: the wage is the purchaser-price
        # index times the target, and that index is built from the prices the
        # wage itself helps determine. Stopping on the price gap alone would
        # leave the wage one iteration behind, and the finite-difference
        # Jacobian would see a jump.
        pva = value_added_price(econ, Wm)
        pqf = 1.0 + ptax_f
        for _ in range(_PRICE_ITERS):
            pa, _, _ = production(pqf, Wm, quantities=False, pva=pva)
            pd_ = econ.msh.T @ pa
            arm, dsh, msh_ = armington(
                econ.dom_share, domestic_price(pd_ * out_wedge, pe), pm, econ.sigma_arm)
            pbase = arm * econ.arm_share + pqf @ econ.mq
            new = pbase * (1 + ptax_f)
            wage_settled = True
            if rigid_wages:
                P_now = _laspeyres_purchaser(econ, pbase * (1.0 + ptax_a))
                for i, target_i in rigid_wages:
                    w_target = target_i * P_now
                    wage_settled = wage_settled and abs(w_target - w[i]) < 1e-14
                    w[i] = w_target
                    Wm[i, :] = w_target
                pva = value_added_price(econ, Wm)
            if np.max(np.abs(new - pqf)) < 1e-14 and wage_settled:
                pqf = new
                break
            pqf = new
        pa, coef, fac_per = production(pqf, Wm, pva=pva)
        # Supply of each imperfectly mobile factor to each activity, and its
        # income: the pools' prices times their endowments, which is also the
        # activity prices times the supplies.
        mob_supply, mob_price, mob_income = {}, {}, {}
        for i, m in mob.items():
            S_o, W_o = m.supply(Wm[i], endow[i])
            mob_supply[i], mob_price[i] = S_o, W_o
            mob_income[i] = float(W_o @ endow[i])
        # Under a transformation between segments, each household's supply to
        # each segment at this period's wages.
        seg_supply = {k: g.supplies(w) for k, g in enumerate(groups)
                      if g.transform is not None}
        pd_ = econ.msh.T @ pa
        p_dom = domestic_price(pd_ * out_wedge, pe)
        arm, dsh, msh_ = armington(econ.dom_share, p_dom, pm, econ.sigma_arm)
        # How output splits between the home market and exports. Under CET the
        # split answers to the export price relative to the domestic one; under
        # fixed shares it is the base-year split and exports fetch the output price.
        if cet_on:
            _, dom_frac, exp_frac = cet(econ.exp_share, p_dom, pe, sigma_cet)
            p_exp = pe
        else:
            dom_frac, exp_frac, p_exp = 1.0 - econ.exp_share, econ.exp_share, pd_ * out_wedge
        pbase = arm * econ.arm_share + pqf @ econ.mq
        pqf = pbase * (1 + ptax_f)
        pq = pbase * (1 + ptax_a)
        # The household carbon tax is specific; at this price it is the ad
        # valorem wedge carb / pq, paid to government like the government wedge.
        Wg = Wg_set + carb_hh[None, :] / np.maximum(pq, 1e-12)[None, :]
        Wt = W + Wg

        # Labour income is the wage times employment, not times the labour force.
        # Under a fixed real wage or a wage curve, employment IS labour demand, and
        # demand depends on the income that employment pays. The map is affine, so
        # two probes give it in closed form. Iterating it instead leaves a residual
        # that the outer finite-difference Jacobian reads as a kink.
        P_level = _laspeyres_purchaser(econ, pq)

        def quantities(paid):
            # --- household budgets, with the wedge recycled in closed form
            transfers = base_transfers - levy * econ.pop_share + gov_tr + remit
            fac_inc = w * paid
            for i in mob:
                fac_inc[i] = mob_income[i]
            gov_fac = float(fac_inc @ fac_gov_sh)
            row_fac = float(fac_inc @ fac_row_sh)
            base_income = (facshare_eff @ (fac_inc * (1.0 - fac_gov_sh - fac_row_sh))
                           + transfers)
            # Each household's workers in a segment group, and what they earn.
            seg_households = []
            for k, g in enumerate(groups):
                e_h, u_h = g.households(paid, U_now[k], seg_supply.get(k))
                idx = g.group.index
                base_income = base_income + e_h @ (w[idx] * (1.0 - fac_gov_sh[idx]
                                                             - fac_row_sh[idx]))
                seg_households.append((e_h, u_h))
            direct = econ.hh_direct @ (pa * act_scale)
            eff_p = pq[None, :] * (1.0 + Wt)
            gamma = econ.hh_committed
            committed_value = (eff_p * gamma).sum(axis=1)

            # Crediting the wedge to income raises spending, which raises the wedge,
            # which raises income -- an infinite series, not a single pass. Demand is
            # affine in spending under both demand systems, so revenue_h = A_h + B_h
            # spend_h exactly and the series is summed in closed form. Iterating it
            # converges but leaves small discontinuities that wreck the outer
            # Newton's finite-difference Jacobian.
            with np.errstate(divide="ignore", invalid="ignore"):
                capture = np.where(1.0 + Wt > 0, W / (1.0 + Wt), 0.0)
                capture_g = np.where(1.0 + Wt > 0, Wg / (1.0 + Wt), 0.0)
            B_cap = (beta_les * capture).sum(axis=1)
            B_gov = (beta_les * capture_g).sum(axis=1)
            A_cap = (gamma * W * pq[None, :]).sum(axis=1) - B_cap * committed_value
            A_gov = (gamma * Wg * pq[None, :]).sum(axis=1) - B_gov * committed_value

            net = (1.0 - dtax) * (1.0 - econ.hh_savrate)
            const_spend = base_income * net - direct
            denom = 1.0 - float((B_cap * net * alpha_recycle).sum())
            wedge_rev = ((A_cap.sum() + float((B_cap * const_spend).sum())) / denom
                         if np.any(W != 0.0) else 0.0)
            hh_income = base_income + wedge_rev * alpha_recycle
            spend = hh_income * net
            hh_com_spend = spend - direct
            if np.any(hh_com_spend < -1e-9):
                bad = [econ.hhs[i] for i in np.where(hh_com_spend < -1e-9)[0]]
                raise ValueError(
                    f"household(s) {bad} spend more on direct purchases from activities than "
                    f"their whole budget. Clipping this to zero would destroy income and break "
                    f"Walras' law silently.")

            # --- demand
            if econ.spec.household_demand == "les":
                supernum = hh_com_spend - committed_value
                hh_q = gamma + beta_les * supernum[:, None] / np.maximum(eff_p, 1e-12)
            else:
                hh_q = econ.hh_budget * hh_com_spend[:, None] / np.maximum(eff_p, 1e-12)
            hh_final = hh_q.sum(axis=0)
            wedge_paid = (hh_q * W * pq[None, :]).sum(axis=1)
            gov_cost = float(-(hh_q * Wg * pq[None, :]).sum()) + sc.government_absorbs

            hh_dtax_paid = hh_income * dtax
            hh_sav = float(((hh_income - hh_dtax_paid) * econ.hh_savrate).sum())
            # Government income the tax block does not see: direct taxes and its own
            # factor income, net of what it transfers to households.
            gov_inst = float(hh_dtax_paid.sum()) + gov_fac - float(gov_tr.sum())

            # --- quantity system. Home demand for the domestic variety, over the
            # CET's home share, is output at its tax-inclusive value; over
            # (1 + tauz0) it is commodity output at producer value.
            with np.errstate(divide="ignore", invalid="ignore"):
                Gp = np.diag(dsh * econ.arm_share / np.maximum(dom_frac * tz0, 1e-12))
            Bm = coef @ econ.msh
            fuel_carbon_per_unit = (carb_fuel * coef).sum(axis=0)
            M = np.linalg.inv(np.eye(nc) - Bm @ Gp - econ.mq)
            other_fixed = coef @ X_exog                   # own-account inputs, firm rate

            def duty_of(a_vec):
                # The tariff and the output tax are levels: each rate times its
                # own base, imports at the border and output at the producer price.
                duty = 0.0
                if tm is not None:
                    duty += float((tm * pm_border * msh_ * econ.arm_share * a_vec / tf0).sum())
                if tauz is not None:
                    duty += float((tauz * pd_ * (Gp @ a_vec)).sum())
                return duty

            def tax_of(a_vec, fin_c):
                firm_part = a_vec - fin_c                 # final demand at the CONSUMER rate
                # The activity tax falls on own-account output too: households pay
                # it in the price of what they buy straight from the activity.
                x_vec = econ.msh @ (Gp @ a_vec) + X_exog
                return (float((pbase * (firm_part * ptax_f + fin_c * ptax_a)).sum())
                        + float((atax * x_vec).sum())
                        + fuel_carbon_per_unit @ x_vec      # carbon paid in the fuel price
                        + duty_of(a_vec))

            level = inv_budget0 if sc.investment_level is None else float(sc.investment_level)
            inv_fixed = inv_share * level / (1.0 + econ.ptax)

            def flows(gov_budget, saving):
                """The circular flow at a given government budget and saving."""
                if target == "saving" and adjusts == "spending":
                    gov_q = gov_w * gov_budget / np.maximum(pq, 1e-12)
                elif closure.government == "real_fixed":
                    gov_q = econ.gov_dem / (1.0 + econ.ptax)
                elif closure.government == "nominal_fixed":
                    gov_q = econ.gov_dem * nu / np.maximum(pq, 1e-12)
                else:
                    gov_q = gov_w * gov_budget / np.maximum(pq, 1e-12)
                # Fixed investment is held in QUANTITIES, at base-year purchaser
                # prices. A mix changes the basket and keeps the base-year real total.
                inv_q = (inv_share * saving / np.maximum(pqf, 1e-12)
                         if closure.investment == "savings_driven" else inv_fixed)
                fin_c = hh_final + gov_q
                A = M @ (fin_c + other_fixed + inv_q)
                gov_income = tax_of(A, fin_c) + gov_inst - gov_cost
                gov_spend = float(pq @ gov_q)
                available = hh_sav + levy + gov_income - gov_spend + foreign_sav
                return gov_q, inv_q, A, gov_income, gov_spend, available

            # Saving funds investment, investment drives output, output drives
            # tax revenue, and revenue is government saving -- or, under a
            # government saving rate, also government spending. Every link is
            # affine in the two budgets, so they are solved from probes in closed
            # form rather than iterated.
            free = []
            if closure.government == "saving_rate" or (target == "saving"
                                                       and adjusts == "spending"):
                free.append("gov")
            if closure.investment == "savings_driven":
                free.append("sav")

            def gap(u):
                g = u[free.index("gov")] if "gov" in free else 0.0
                sv = u[free.index("sav")] if "sav" in free else 0.0
                _, _, _, inc, _, avail = flows(g, sv)
                out_g = []
                if "gov" in free:
                    if closure.government == "saving_rate":
                        out_g.append(g - (1.0 - gov_savrate) * inc)
                    else:                                  # spend all but the target saving
                        out_g.append(g - (inc - target_value))
                if "sav" in free:
                    out_g.append(sv - avail)
                return np.array(out_g)

            u = np.zeros(len(free))
            if free:
                r0 = gap(u)
                J = np.empty((len(free), len(free)))
                for k in range(len(free)):
                    e = np.zeros(len(free))
                    e[k] = sav_scale
                    J[:, k] = (gap(e) - r0) / sav_scale
                u = np.linalg.solve(J, -r0)
            g_now = u[free.index("gov")] if "gov" in free else 0.0
            s_now = u[free.index("sav")] if "sav" in free else 0.0
            gov_q, inv_q, A, gov_income, gov_spend, available = flows(g_now, s_now)
            sav_gap = None
            if closure.investment == "savings_driven":
                savings = s_now
            else:
                savings = float(pqf @ inv_q)
                sav_gap = (available - savings) / sav_scale

            Q = Gp @ A
            X = econ.msh @ Q + X_exog
            # The government account at the solution, instrument by instrument,
            # from the same formulas the closure solved with.
            fin_c = hh_final + gov_q
            gov_account = dict(
                product_tax=float((pbase * ((A - fin_c) * ptax_f + fin_c * ptax_a)).sum()),
                activity_tax=float((atax_set * X).sum()),
                carbon_production=float(carb_act @ X + (carb_fuel * coef * X[None, :]).sum()),
                carbon_households=float(hh_q.sum(axis=0) @ carb_hh),
                tariff=(0.0 if tm is None else
                        float((tm * pm_border * msh_ * econ.arm_share * A / tf0).sum())),
                output_tax=0.0 if tauz is None else float((tauz * pd_ * Q).sum()),
                direct_tax=float(hh_dtax_paid.sum()), factor_income=gov_fac,
                wedge=float((hh_q * Wg_set * pq[None, :]).sum()),   # negative for a refund
                transfers=float(gov_tr.sum()), consumption=gov_spend,
                absorbed=float(sc.government_absorbs))
            gov_account["revenue"] = sum(gov_account[k] for k in (
                "product_tax", "activity_tax", "carbon_production", "carbon_households",
                "tariff", "output_tax", "direct_tax", "factor_income", "wedge"))
            gov_account["saving"] = (gov_account["revenue"] - gov_account["transfers"]
                                     - gov_account["consumption"] - gov_account["absorbed"])

            arm_q = econ.arm_share * A
            imports = msh_ * arm_q / tf0                  # at the border
            exports = exp_frac * tz0 * Q
            fac_by_act = fac_per * X[None, :]
            fac_dem = fac_by_act.sum(axis=1)
            # Each imperfectly mobile factor clears activity by activity. Its
            # endowment is what `factor_demand` reports, and its price is its
            # income over that endowment.
            w_report = w.copy()
            mob_gaps, mob_out = {}, []
            for i, m in mob.items():
                g = ((fac_by_act[i] - mob_supply[i].sum(axis=0))
                     / np.maximum(np.abs(base_use[i]), 1.0))[m.users]
                mob_out.extend(g)
                mob_gaps[i] = float(g[np.argmax(np.abs(g))])
                total = float(endow[i].sum())
                w_report[i] = mob_income[i] / total if total > 0 else nu
                fac_dem[i] = total
            trade = float((pm_border * imports).sum() - (p_exp * exports).sum())
            # The current account: trade plus factor income paid abroad, less net
            # transfers received from abroad. It is this, not trade alone, that
            # foreign saving finances.
            current = trade + row_fac - float(remit.sum())

            reference = supply.copy()
            if closure.labour == "elastic_supply":
                reference[lab] = paid[lab]
            # A segment's market is against what it pays: its employment, and for
            # the informal one the rest of the labour force.
            for g in groups:
                reference[g.group.index] = paid[g.group.index]
            gaps = (fac_dem - reference) / np.maximum(np.abs(reference), 1.0)
            # The numeraire market stays the Walras check against its SUPPLY. A labour
            # closure that changed this denominator would make the check an identity.
            gaps[num] = (fac_dem[num] - supply[num]) / max(abs(float(supply[num])), 1.0)
            for i, g in mob_gaps.items():
                gaps[i] = g
            out = []
            for i in others:
                if closure.labour == "wage_curve" and i == lab:
                    force = float(supply[lab]) / (1.0 - closure.natural_unemployment)
                    u = 1.0 - float(fac_dem[lab]) / force
                    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
                    out.append(np.log(w[lab] / P_level)
                               + closure.wage_curve_elasticity * np.log(u / closure.natural_unemployment))
                else:
                    out.append(float(gaps[i]))
            if closure.solves_exchange_rate:
                out.append((current - foreign_sav) / ext_scale)
            else:
                # The exchange rate is exogenous, so the current account is whatever
                # it turns out to be and foreign saving must equal it. Holding
                # foreign saving at its base value instead leaves the shortfall
                # nowhere, and it surfaces as a Walras residual.
                out.append((current - foreign_sav) / ext_scale)
            if sav_gap is not None:
                out.append(sav_gap)
            # The curves, in logs: log(E/E0) = -η log(pWe/shift), and the mirror
            # for imports. A volume driven to zero gives a large finite residual.
            out.extend(np.log(np.maximum(exports[curve_x], 1e-300) / econ.exp0[curve_x])
                       + eta_x[curve_x] * log_px)
            out.extend(np.log(np.maximum(imports[curve_m], 1e-300) / econ.imp0[curve_m])
                       - psi_m[curve_m] * log_pm)
            if tax_lever:
                hit = (gov_account["saving"] if target == "saving"
                       else gov_account["revenue"])
                out.append((hit - target_value) / sav_scale)
            out.extend(mob_out)
            # Harris–Todaro: the informal wage is the expected wage of queueing
            # for a rigid job.
            for k in i_ht:
                g = groups[k]
                E_R = fac_dem[g.rigid]
                expected = g.lam * float(w[g.rigid] @ E_R) / (float(E_R.sum()) + U_now[k])
                out.append(float(np.log(w[g.informal]) - np.log(max(expected, 1e-300))))
            markets = {}
            for k, g in enumerate(groups):
                idx = g.group.index
                e_h, u_h = seg_households[k]
                if g.transform is None:
                    seg_sup = fac_dem[idx].copy()
                    queue = np.zeros(len(idx))
                    U_k = U_now[k]
                    E_R = float(fac_dem[g.rigid].sum())
                    rate = U_k / (E_R + U_k) if E_R + U_k > 0 else 0.0
                else:
                    seg_sup = seg_supply[k].sum(axis=0)
                    queue = np.where(np.isin(idx, g.rigid), seg_sup - fac_dem[idx], 0.0)
                    U_k = float(queue.sum())
                    S_R = float(seg_sup[np.isin(idx, g.rigid)].sum())
                    rate = U_k / S_R if S_R > 0 else 0.0
                markets[g.group.name] = dict(
                    segments=list(g.group.factors),
                    informal=econ.factors[g.informal] if g.informal >= 0 else None,
                    wage=w[idx].copy(), real_wage=w[idx] / P_level,
                    employment=fac_dem[idx].copy(), supply=seg_sup, queue=queue,
                    labour_force=g.N, unemployment=U_k, unemployment_rate=rate,
                    harris_todaro=g.harris_todaro,
                    household_workforce=g.workforce.copy(),
                    household_employment=e_h.copy(), household_unemployed=u_h.copy())

            p_base_yard = 1.0 + econ.ptax          # constant-price yardstick, scenario independent
            gov_sav = savings - hh_sav - levy - foreign_sav
            lam = float(p_base_yard @ inv_q) / savings if abs(savings) > 1e-12 else 0.0
            if lab is None:
                employment = float("nan")
                unemployment = 0.0
                real_wage_now = float("nan")
            else:
                real_wage_now = float(w_report[lab] / P_level)
                employment = float(fac_dem[lab])
                if closure.labour == "wage_curve":
                    force = float(supply[lab]) / (1.0 - closure.natural_unemployment)
                    unemployment = force - employment
                elif closure.labour == "fixed_real_wage":
                    unemployment = float(supply[lab]) - employment
                else:
                    unemployment = 0.0
            Wm_report = Wm.copy()
            for i, m in mob.items():
                Wm_report[i, ~m.users] = w_report[i]
            state.update(
                w=w_report, w_by_act=Wm_report,
                markets=markets or None,
                pools={econ.factors[i]: dict(pools=list(m.pools), price=mob_price[i].copy(),
                                             endowment=endow[i].copy(),
                                             supply=mob_supply[i].copy())
                       for i, m in mob.items()} or None,
                er=er, pa=pa, pd=pd_, pq=pq, pqf=pqf, pbase=pbase, Q=Q, X=X, A=A,
                hh_q=hh_q, hh_income=hh_income, hh_com_spend=hh_com_spend, direct=direct,
                wedge_paid=wedge_paid, wedge_rev=wedge_rev, gov_cost=gov_cost,
                gov_q=gov_q, inv_q=inv_q, savings=savings, fac_dem=fac_dem,
                fac_by_act=fac_by_act,
                imports=imports, exports=exports, trade=trade, gaps=gaps,
                world_pe=world_pe_now, world_pm=world_p_now,
                dom_sales=dsh * econ.arm_share * A, p_dom=p_dom,
                walras=float(gaps[num]), levy=levy, foreign_sav=foreign_sav, pe=p_exp,
                hh_dtax_paid=hh_dtax_paid, row_fac=row_fac, gov_fac=gov_fac,
                gov_account=gov_account, hh_sav=hh_sav, tax_scale=tax_scale,
                io_used=coef, va_per=fac_per.sum(axis=0),
                real=dict(hh_real=float(p_base_yard @ hh_q.sum(axis=0)),
                          gov_real=float(p_base_yard @ gov_q),
                          inv_real=float(p_base_yard @ inv_q),
                          hh_sav_real=lam * hh_sav, gov_sav_real=lam * gov_sav,
                          for_sav_real=lam * foreign_sav,
                          hh_real_income=hh_com_spend / _price_index(econ, eff_p, beta_les),
                          hh_price_index=_price_index(econ, eff_p, beta_les),
                          hh_committed_value=committed_value),
            employment=employment, unemployment=unemployment,
            real_wage=real_wage_now)

            return fac_dem[slack_idx], out

        def paid_of(x):
            paid = supply.copy()
            if closure.labour == "elastic_supply":
                paid[lab] = supply[lab] * (w[lab] / P_level) ** closure.labour_supply_elasticity
            paid[slack_idx] = x
            # The informal segment employs the rest of the labour force.
            for k, g in enumerate(groups):
                if g.transform is not None:
                    # A flexible segment pays its whole supply; it clears against it.
                    tot = seg_supply[k].sum(axis=0)
                    for i in g.clearing:
                        paid[i] = tot[g.group.index.index(i)]
                else:
                    paid[g.informal] = g.N - float(paid[g.hired].sum()) - U_now[k]
            return paid

        if slack_idx:
            # Employment that is demand, D(x) = d0 + J (x - x0), is affine in the
            # employment that pays the income; k + 1 probes give it and the fixed
            # point x = D(x) is one linear solve.
            x0 = supply[slack_idx].astype(float)
            steps = np.maximum(np.abs(x0), 1.0) * 1e-3
            d0, _ = quantities(paid_of(x0))
            J = np.empty((len(slack_idx), len(slack_idx)))
            for k in range(len(slack_idx)):
                x = x0.copy()
                x[k] += steps[k]
                J[:, k] = (quantities(paid_of(x))[0] - d0) / steps[k]
            A_fix = np.eye(len(slack_idx)) - J
            if np.linalg.cond(A_fix) > 1e12:
                raise RuntimeError(
                    "labour demand responds one-for-one to the wage bill, so employment "
                    "is not determined. That is a closure failure, not a solution.")
            _, out = quantities(paid_of(np.linalg.solve(A_fix, d0 - J @ x0)))
        else:
            _, out = quantities(paid_of(np.zeros(0)))
        return out

    # Convergence is judged on the residual, never on the solver's success flag:
    # `hybr` reports failure whenever its finite-difference Jacobian stops
    # improving, which includes the case where the start point already IS the
    # solution, and it terminates early from a warm start that sits exactly on
    # the previous period's answer.
    attempts = []
    if warm_start is not None:
        z0 = warm_start.z if isinstance(warm_start, PeriodResult) else np.asarray(warm_start)
        z0 = np.asarray(z0, dtype=float)
        if z0.size == n_unknown:
            attempts.append(("warm", z0, "hybr"))
    attempts += [("cold", np.zeros(n_unknown), "hybr"),
                 ("perturbed", np.full(n_unknown, 1e-4), "hybr"),
                 ("lm", np.zeros(n_unknown), "lm")]

    best = None
    for label, start, method in attempts:
        try:
            sol = root(residuals, start, method=method,
                       options={"xtol": 1e-14} if method == "hybr"
                       else {"xtol": 1e-14, "ftol": 1e-14})
        except ValueError:
            continue
        res = float(np.max(np.abs(residuals(sol.x))))
        if best is None or res < best[0]:
            best = (res, sol.x, label, sol.message)
        if res < closure.tolerance:
            break
    if best is None:
        raise RuntimeError("solve_period: every start point failed before a residual was formed")

    res, x, label, message = best
    residuals(x)
    ok = res < closure.tolerance
    for name, mk in (state["markets"] or {}).items():
        short = [f for f, q in zip(mk["segments"], mk["queue"]) if q < -1e-9]
        if short and ok:
            raise ValueError(
                f"labour group {name!r}: rigid segments {short} would hire more than the "
                f"workers who supply them at their wage, a negative queue. The rigid wage is "
                f"too low for this shock, or the base-year queue rate too small.")
        if mk["informal"] is None:
            continue
        if mk["unemployment"] < -1e-9 and ok:
            raise ValueError(
                f"labour group {name!r}: Harris–Todaro needs negative unemployment here, "
                f"{mk['unemployment']:.6g}. The informal wage is above what queueing for a "
                f"rigid job is expected to pay, so nobody would queue: the rigid wage is too "
                f"low for this shock, or the market is dual.")
        k_b = mk["segments"].index(mk["informal"])
        short = np.where(mk["household_employment"][:, k_b] < -1e-9)[0]
        if short.size and ok:
            raise ValueError(
                f"labour group {name!r}: households {[econ.hhs[h] for h in short]} would hold "
                f"more rigid jobs and queue places than they have workers, so their informal "
                f"employment is negative. The rigid wage or the shock is outside what the "
                f"group's labour force can supply.")
    if not ok and raise_on_fail:
        raise RuntimeError(
            f"solve_period: residual {res:.3e} above tolerance {closure.tolerance:.1e} "
            f"after {len(attempts)} start points. Solver message: {message}")

    return PeriodResult(
        econ=econ, closure=closure, scenario=sc, z=np.asarray(x, dtype=float),
        factor_prices=state["w"], exchange_rate=state["er"], pa=state["pa"],
        pd=state["pd"], pq=state["pq"], pqf=state["pqf"], pbase=state["pbase"],
        Q=state["Q"], X=state["X"], absorption=state["A"], imports=state["imports"],
        exports=state["exports"], hh_quantity=state["hh_q"], hh_income=state["hh_income"],
        hh_commodity_spend=state["hh_com_spend"], hh_direct_outlay=state["direct"],
        hh_wedge_paid=state["wedge_paid"], gov_quantity=state["gov_q"],
        investment_quantity=state["inv_q"], factor_demand=state["fac_dem"],
        factor_demand_by_activity=state["fac_by_act"],
        savings=state["savings"], wedge_revenue=state["wedge_rev"],
        government_cost=state["gov_cost"], real=state["real"],
        gaps={f: float(g) for f, g in zip(econ.factors, state["gaps"])},
        walras=state["walras"], residual=res, converged=ok, attempt=label, message=message,
        pe=state["pe"], hh_direct_tax=state["hh_dtax_paid"],
        factor_income_abroad=state["row_fac"], foreign_saving=state["foreign_sav"],
        employment=state["employment"], unemployment=state["unemployment"],
        real_wage=state["real_wage"], world_export_price=state["world_pe"],
        world_import_price=state["world_pm"], domestic_sales=state["dom_sales"],
        p_domestic=state["p_dom"], government=state["gov_account"],
        household_saving=state["hh_sav"], forced_saving=state["levy"],
        tax_scale=state["tax_scale"], input_coefficients=state["io_used"],
        factor_prices_by_activity=state["w_by_act"], factor_pools=state["pools"],
        labour_markets=state["markets"],
        value_added_per_unit=state["va_per"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _energy_index(theta, prices, sigma):
    """CES price of the energy bundle by activity, fuel prices differing by activity."""
    sigma = np.broadcast_to(np.asarray(sigma, float), theta.shape[1:])
    out = np.ones(theta.shape[1])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        unit = np.abs(sigma - 1.0) < 1e-9
        logs = np.where(theta > 0, theta * np.log(np.where(theta > 0, prices, 1.0)), 0.0)
        cd = np.exp(logs.sum(axis=0))
        terms = np.where(theta > 0, theta * np.where(theta > 0, prices, 1.0)
                         ** (1.0 - sigma)[None, :], 0.0)
        ces = terms.sum(axis=0) ** (1.0 / (1.0 - np.where(unit, 0.5, sigma)))
    return np.where(unit, cd, ces) * out


def _two_input_ces(a, p1, b, p2, sigma):
    """CES price of two inputs with value shares ``a`` and ``b``.

    With ``sigma == 0`` it is linear, ``a p1 + b p2``, which also holds for a
    negative share (an activity with negative value added on fixed proportions).
    """
    sigma = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        unit = np.abs(sigma - 1.0) < 1e-9
        cd = np.power(p1, a) * np.power(p2, b)
        s = np.where(unit, 0.5, sigma)
        ces = (a * np.power(p1, 1.0 - s) + b * np.power(p2, 1.0 - s)) ** (1.0 / (1.0 - s))
    return np.where(sigma == 0, a * p1 + b * p2, np.where(unit, cd, ces))


def _ratio_power(num, den, sigma):
    """``(num / den) ** sigma``, one whenever ``sigma`` is zero."""
    sigma = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.power(num / den, sigma)
    return np.where(sigma == 0, 1.0, out)


def _trade_curves(econ: Economy):
    """Elasticities and the commodities on an export-demand or import-supply curve."""
    spec = econ.spec
    eta_x = spec.export_demand_vector(econ.coms)
    psi_m = spec.import_supply_vector(econ.coms)
    if np.isfinite(eta_x).any() and spec.export_rule != "cet":
        raise ValueError("export_demand is read only under export_rule='cet'")
    curve_x = np.where(np.isfinite(eta_x) & (econ.exp0 > 0))[0]
    curve_m = np.where(np.isfinite(psi_m) & (econ.imp0 > 0))[0]
    return eta_x, psi_m, curve_x, curve_m


def _marginal_budget_shares(econ: Economy) -> np.ndarray:
    """Marginal budget shares: the LES betas, or the Cobb-Douglas shares themselves."""
    if econ.spec.household_demand != "les":
        return econ.hh_budget
    # Recovered from the calibration: p*gamma = p*q - beta*Y*, so beta = (p*q - p*gamma)/Y*.
    Ystar = -1.0 / econ.spec.frisch
    value_q = econ.hh_budget                                  # shares of total outlay
    committed = econ.hh_committed * (1.0 + econ.ptax)[None, :]
    # The LES was calibrated on commodity spending. Income net of saving equals it
    # only with no direct tax, no transfers abroad and no purchases made straight
    # from an activity, so the calibrated budget is used when it is known.
    total = np.maximum(econ.hh_spend0 if econ.hh_spend0 is not None
                       else econ.hh_income0 * (1 - econ.hh_savrate), 1e-12)
    share_committed = committed / total[:, None]
    beta = (value_q - share_committed) / Ystar
    beta = np.maximum(beta, 0.0)
    return beta / np.maximum(beta.sum(axis=1, keepdims=True), 1e-12)


def _laspeyres_purchaser(econ: Economy, pq: np.ndarray) -> float:
    """Base-weighted index of purchaser prices. Equals the numeraire at the base year.

    Weights are base-year household quantities, so the basket does not follow the
    scenario. A real wage is this index in the denominator: a yardstick that moved
    with the policy's own basket would make the wage rule a different rule in
    every scenario.
    """
    value0 = (econ.hh_budget * np.asarray(econ.hh_spend0, float)[:, None]).sum(axis=0)
    pq0 = 1.0 + econ.ptax
    q0 = value0 / np.maximum(pq0, 1e-12)
    return float(pq @ q0) / float(pq0 @ q0)


def _price_index(econ: Economy, eff_p: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Cost-of-living index per household: the exact CD index, the LES marginal one."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(beta > 0, beta * np.log(np.maximum(eff_p, 1e-300)), 0.0)
    return np.exp(logs.sum(axis=1))


def _recycle_shares(econ: Economy, sc: Scenario, facshare: np.ndarray,
                    supply: np.ndarray) -> np.ndarray:
    """Shares in which recycled wedge revenue reaches households."""
    rule = sc.wedge_revenue_to
    if callable(rule):
        out = np.asarray(rule(econ), dtype=float)
    elif rule == "lumpsum":
        out = econ.pop_share.copy()
    elif rule == "capital" or isinstance(rule, (int, float)):
        fname = sc.wedge_revenue_factor or econ.factors[-1]
        fi = econ.factor(fname)
        inc = facshare[:, fi] * supply[fi]
        owner = inc / max(float(inc.sum()), 1e-12)
        if rule == "capital":
            out = owner
        else:
            lam = float(rule)
            if not 0.0 <= lam <= 1.0:
                raise ValueError(f"the recycling weight must lie in [0, 1], got {lam}")
            out = lam * owner + (1.0 - lam) * econ.pop_share
    else:
        raise ValueError(f"unknown wedge_revenue_to: {rule!r}")
    if out.shape != (econ.n_hh,):
        raise ValueError(f"recycling shares must have {econ.n_hh} entries, got {out.shape}")
    return out
