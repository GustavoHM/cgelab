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
* **Closure.** See :class:`cgekit.closure.Closure`.

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
from .scenario import Scenario, investment_shares

__all__ = ["PeriodResult", "solve_period", "value_added_price", "factor_intensity"]

_PRICE_ITERS = 200
_RECYCLE_TOL = 1e-13


# ---------------------------------------------------------------------------
# Price blocks
# ---------------------------------------------------------------------------

def value_added_price(econ: Economy, w: np.ndarray) -> np.ndarray:
    """Unit cost index of value added, by activity."""
    if econ.spec.value_added == "cobb_douglas":
        nest = cobb_douglas_price(econ.theta, w)
    else:
        nest = ces_price(econ.theta, w, econ.sigma_va)
    leo = w @ econ.aunit
    with np.errstate(divide="ignore", invalid="ignore"):
        leo_unit = np.where(econ.vunit != 0, leo / np.where(econ.vunit == 0, 1.0, econ.vunit), 0.0)
    return np.where(econ.leontief, leo_unit, nest)


def factor_intensity(econ: Economy, w: np.ndarray, *, vunit=None, aunit=None) -> np.ndarray:
    """Factor input per unit of value added, ``(n_fac, n_act)``.

    ``vunit`` and ``aunit`` override the calibrated coefficients, which is how a
    productivity path enters: the CES branch is scale-free and unaffected, but
    the fixed-proportions branch is a ratio of the two and must see both.
    """
    vunit = econ.vunit if vunit is None else vunit
    aunit = econ.aunit if aunit is None else aunit
    pva = value_added_price(econ, w)
    if econ.spec.value_added == "cobb_douglas":
        with np.errstate(divide="ignore", invalid="ignore"):
            nest = np.where(econ.theta > 0, econ.theta * pva[None, :] / w[:, None], 0.0)
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
    # A fixed real wage is not an unknown: it is the price index times the target.
    # Dropping it without dropping the market equation would overdetermine the model.
    others = [i for i in range(nf) if i != num
              and not (closure.labour == "fixed_real_wage" and i == lab)]
    i_ext = len(others)                       # external-closure unknown
    i_inv = i_ext + 1                         # forced-saving unknown, when present
    n_unknown = i_ext + 1 + (1 if closure.investment == "fixed_real" else 0)

    supply = econ.fac_supply if sc.factor_supply is None else np.asarray(sc.factor_supply, float)
    facshare = (econ.hh_facshare if sc.factor_ownership is None
                else np.asarray(sc.factor_ownership, float))
    tfp = np.ones(na) if sc.productivity is None else np.asarray(sc.productivity, float)
    ptax_a = econ.ptax if sc.product_tax is None else np.asarray(sc.product_tax, float)
    ptax_f = ptax_a if sc.product_tax_firm is None else np.asarray(sc.product_tax_firm, float)
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
    W = np.zeros((nh, nc)) if sc.consumer_wedge is None else np.asarray(sc.consumer_wedge, float)
    Wg = (np.zeros((nh, nc)) if sc.government_wedge is None
          else np.asarray(sc.government_wedge, float))
    Wt = W + Wg
    base_transfers = np.zeros(nh) if sc.transfers is None else np.asarray(sc.transfers, float)
    dtax = econ.zeros_if_none("hh_dtax", nh)
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
    trade_target = (econ.foreign_sav + float(sum(sc.fdi.values()))) * nu
    gov_tr = econ.zeros_if_none("hh_gov_transfer", nh) * nu     # fixed in numeraire units,
    remit = econ.zeros_if_none("hh_remit_net", nh) * nu         # like foreign saving
    atax = econ.atax * nu
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
    vunit_eff = econ.vunit / tfp
    aunit_eff = econ.aunit / tfp[None, :]

    X_exog = econ.X0 * act_scale * exog
    exog_intermediate = io_eff @ X_exog           # this block's own input demand

    alpha_recycle = _recycle_shares(econ, sc, facshare, supply)
    beta_les = _marginal_budget_shares(econ)
    inv_share = investment_shares(econ, sc)
    inv_budget0 = float(econ.inv_dem.sum())
    state: dict = {}

    def domestic_price(pz, pe):
        """Price of domestic sales: the output price itself, or its CET split."""
        return cet_domestic_price(econ.exp_share, pz, pe, sigma_cet) if cet_on else pz

    def residuals(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        w = np.full(nf, nu)
        for k, i in enumerate(others):
            w[i] = nu * float(np.exp(z[k]))
        if closure.solves_exchange_rate:
            er = float(np.exp(z[i_ext]))
            foreign_sav = trade_target
        else:
            er = closure.exchange_rate
            foreign_sav = trade_target + float(z[i_ext]) * ext_scale
        levy = float(z[i_inv]) * sav_scale if closure.investment == "fixed_real" else 0.0
        # Border price is what the balance of payments pays. The Armington import
        # price carries the tariff, normalised so that at the calibrated rate it
        # is 1, the same unit the rest of the base year uses. An output tax is
        # the same kind of normalisation on the domestic price.
        pm_border = er * world_p
        if tm is None:
            pm = pm_border
        else:
            pm = pm_border * (1.0 + tm) / (1.0 + tm0)
        if tauz is None:
            out_wedge = 1.0
        else:
            out_wedge = (1.0 + tauz) / (1.0 + tauz0)
        pe = er * world_pe

        # --- prices: a fixed point, because margins are themselves delivered goods.
        # A fixed real wage joins the same loop: the wage is the purchaser-price
        # index times the target, and that index is built from the prices the
        # wage itself helps determine. Stopping on the price gap alone would
        # leave the wage one iteration behind, and the finite-difference
        # Jacobian would see a jump.
        pva = value_added_price(econ, w)
        pqf = 1.0 + ptax_f
        real_wage_target = 1.0 if closure.real_wage is None else float(closure.real_wage)
        for _ in range(_PRICE_ITERS):
            pa = io_eff.T @ pqf + pva * vunit_eff + atax
            pd_ = econ.msh.T @ pa
            arm, dsh, msh_ = armington(
                econ.dom_share, domestic_price(pd_, pe) * out_wedge, pm, econ.sigma_arm)
            pbase = arm * econ.arm_share + pqf @ econ.mq
            new = pbase * (1 + ptax_f)
            wage_settled = True
            if closure.labour == "fixed_real_wage":
                P_now = _laspeyres_purchaser(econ, pbase * (1.0 + ptax_a))
                w_target = real_wage_target * P_now
                wage_settled = abs(w_target - w[lab]) < 1e-14
                w[lab] = w_target
                pva = value_added_price(econ, w)
            if np.max(np.abs(new - pqf)) < 1e-14 and wage_settled:
                pqf = new
                break
            pqf = new
        pa = io_eff.T @ pqf + pva * vunit_eff + atax
        pd_ = econ.msh.T @ pa
        p_dom = domestic_price(pd_, pe)
        arm, dsh, msh_ = armington(econ.dom_share, p_dom * out_wedge, pm, econ.sigma_arm)
        # How output splits between the home market and exports. Under CET the
        # split answers to the export price relative to the domestic one; under
        # fixed shares it is the base-year split and exports fetch the output price.
        if cet_on:
            _, dom_frac, exp_frac = cet(econ.exp_share, p_dom, pe, sigma_cet)
            p_exp = pe
        else:
            dom_frac, exp_frac, p_exp = 1.0 - econ.exp_share, econ.exp_share, pd_
        pbase = arm * econ.arm_share + pqf @ econ.mq
        pqf = pbase * (1 + ptax_f)
        pq = pbase * (1 + ptax_a)

        # Labour income is the wage times employment, not times the labour force.
        # Under a fixed real wage or a wage curve, employment IS labour demand, and
        # demand depends on the income that employment pays. The map is affine, so
        # two probes give it in closed form. Iterating it instead leaves a residual
        # that the outer finite-difference Jacobian reads as a kink.
        P_level = _laspeyres_purchaser(econ, pq)
        slack = closure.labour in ("fixed_real_wage", "wage_curve")

        def quantities(paid):
            # --- household budgets, with the wedge recycled in closed form
            transfers = base_transfers - levy * econ.pop_share + gov_tr + remit
            fac_inc = w * paid
            gov_fac = float(fac_inc @ fac_gov_sh)
            row_fac = float(fac_inc @ fac_row_sh)
            base_income = facshare @ (fac_inc * (1.0 - fac_gov_sh - fac_row_sh)) + transfers
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

            if closure.government == "real_fixed":
                gov_q = econ.gov_dem / (1.0 + econ.ptax)
            else:
                gov_q = econ.gov_dem * nu / np.maximum(pq, 1e-12)
            gov_spend = float(pq @ gov_q)
            hh_dtax_paid = hh_income * dtax
            hh_sav = float(((hh_income - hh_dtax_paid) * econ.hh_savrate).sum())
            # Government income the tax block does not see: direct taxes and its own
            # factor income, net of what it transfers to households.
            gov_inst = float(hh_dtax_paid.sum()) + gov_fac - float(gov_tr.sum())

            # --- quantity system
            with np.errstate(divide="ignore", invalid="ignore"):
                Gp = np.diag(dsh * econ.arm_share / np.maximum(dom_frac, 1e-12))
            Bm = io_eff @ econ.msh
            M = np.linalg.inv(np.eye(nc) - Bm @ Gp - econ.mq)
            inv_w = inv_share
            fin_c = hh_final + gov_q                      # taxed at the CONSUMER rate
            other_fixed = exog_intermediate               # taxed at the firm rate

            def duty_of(a_vec):
                # Only the change from the calibrated rate enters government
                # income. At the calibrated rate the power of the tax is 1 and
                # this is zero, so the base year stays the one product tax and
                # the activity tax already closed.
                duty = 0.0
                if tm is not None:
                    duty += float((((tm - tm0) / (1.0 + tm0)) * pm_border
                                   * msh_ * econ.arm_share * a_vec).sum())
                if tauz is not None:
                    # Domestic sales only. Exports fetch the producer price, so
                    # they are not in this wedge; taxing Q here books a receipt
                    # the foreign buyer did not pay.
                    duty += float((((tauz - tauz0) / (1.0 + tauz0))
                                   * p_dom * dsh * econ.arm_share * a_vec).sum())
                return duty

            def tax_of(a_vec):
                firm_part = a_vec - fin_c
                return (float((pbase * (firm_part * ptax_f + fin_c * ptax_a)).sum())
                        + float((atax * (econ.msh @ (Gp @ a_vec))).sum())
                        + duty_of(a_vec))

            sav_gap = None
            if closure.investment == "savings_driven":
                # Saving funds investment, investment drives output, output drives tax
                # revenue and revenue is government saving. Affine in saving, so it is
                # solved in closed form rather than iterated.
                A_const = M @ (fin_c + other_fixed)
                A_slope = M @ (inv_w / np.maximum(pqf, 1e-12))
                c1 = (float((pbase * A_slope * ptax_f).sum())
                      + float((atax * (econ.msh @ (Gp @ A_slope))).sum())
                      + duty_of(A_slope))
                c0 = hh_sav + levy + tax_of(A_const) + gov_inst - gov_spend - gov_cost + foreign_sav
                savings = c0 / (1.0 - c1)
                inv_q = inv_w * savings / np.maximum(pqf, 1e-12)
                A = M @ (fin_c + other_fixed + inv_q)
            else:
                # Fixed in QUANTITIES, at base-year purchaser prices. A mix changes
                # the basket and keeps the base-year real total.
                inv_q = inv_share * inv_budget0 / (1.0 + econ.ptax)
                A = M @ (fin_c + other_fixed + inv_q)
                required = float(pqf @ inv_q)
                available = (hh_sav + levy + tax_of(A) + gov_inst - gov_spend - gov_cost
                             + foreign_sav)
                savings = required
                sav_gap = (available - required) / sav_scale

            Q = Gp @ A
            X = econ.msh @ Q + X_exog

            arm_q = econ.arm_share * A
            imports = msh_ * arm_q
            exports = exp_frac * Q
            fac_by_act = (factor_intensity(econ, w, vunit=vunit_eff, aunit=aunit_eff)
                          * (vunit_eff * X)[None, :])
            fac_dem = fac_by_act.sum(axis=1)
            trade = float((pm_border * imports).sum() - (p_exp * exports).sum())
            # The current account: trade plus factor income paid abroad, less net
            # transfers received from abroad. It is this, not trade alone, that
            # foreign saving finances.
            current = trade + row_fac - float(remit.sum())

            reference = supply.copy()
            if closure.labour == "elastic_supply":
                reference[lab] = paid[lab]
            gaps = (fac_dem - reference) / np.maximum(np.abs(reference), 1.0)
            # The numeraire market stays the Walras check against its SUPPLY. A labour
            # closure that changed this denominator would make the check an identity.
            gaps[num] = (fac_dem[num] - supply[num]) / max(abs(float(supply[num])), 1.0)
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
                out.append((current - trade_target) / ext_scale)
            else:
                # The exchange rate is exogenous, so the current account is whatever
                # it turns out to be and foreign saving must equal it. Holding
                # foreign saving at its base value instead leaves the shortfall
                # nowhere, and it surfaces as a Walras residual.
                out.append((current - foreign_sav) / ext_scale)
            if sav_gap is not None:
                out.append(sav_gap)

            p_base_yard = 1.0 + econ.ptax          # constant-price yardstick, scenario independent
            gov_sav = savings - hh_sav - levy - foreign_sav
            lam = float(p_base_yard @ inv_q) / savings if abs(savings) > 1e-12 else 0.0
            if lab is None:
                employment = float("nan")
                unemployment = 0.0
                real_wage_now = float("nan")
            else:
                real_wage_now = float(w[lab] / P_level)
                employment = float(fac_dem[lab])
                if closure.labour == "wage_curve":
                    force = float(supply[lab]) / (1.0 - closure.natural_unemployment)
                    unemployment = force - employment
                elif closure.labour == "fixed_real_wage":
                    unemployment = float(supply[lab]) - employment
                else:
                    unemployment = 0.0
            state.update(
                w=w, er=er, pa=pa, pd=pd_, pq=pq, pqf=pqf, pbase=pbase, Q=Q, X=X, A=A,
                hh_q=hh_q, hh_income=hh_income, hh_com_spend=hh_com_spend, direct=direct,
                wedge_paid=wedge_paid, wedge_rev=wedge_rev, gov_cost=gov_cost,
                gov_q=gov_q, inv_q=inv_q, savings=savings, fac_dem=fac_dem,
                fac_by_act=fac_by_act,
                imports=imports, exports=exports, trade=trade, gaps=gaps,
                walras=float(gaps[num]), levy=levy, foreign_sav=foreign_sav, pe=p_exp,
                hh_dtax_paid=hh_dtax_paid, row_fac=row_fac, gov_fac=gov_fac,
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

            return (float(fac_dem[lab]) if lab is not None else 0.0), out

        if slack:
            base_L = float(supply[lab])
            step = max(abs(base_L), 1.0) * 1e-3

            def paid_of(level):
                paid = supply.copy()
                paid[lab] = level
                return paid

            d0, _ = quantities(paid_of(base_L))
            d1, _ = quantities(paid_of(base_L + step))
            slope = (d1 - d0) / step
            intercept = d0 - slope * base_L
            if abs(1.0 - slope) < 1e-8:
                raise RuntimeError(
                    "labour demand responds one-for-one to the wage bill, so employment "
                    "is not determined. That is a closure failure, not a solution.")
            _, out = quantities(paid_of(intercept / (1.0 - slope)))
        else:
            paid = supply.copy()
            if closure.labour == "elastic_supply":
                paid[lab] = supply[lab] * (w[lab] / P_level) ** closure.labour_supply_elasticity
            _, out = quantities(paid)
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
        real_wage=state["real_wage"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
