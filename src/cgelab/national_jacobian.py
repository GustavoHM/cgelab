"""The exact Jacobian of the national residual, by forward-mode differentiation.

The national solver's unknowns are the factor prices, the unknown of the external
closure and, under fixed real investment, the levy that finances it; a commodity
on an export-demand or import-supply curve adds its world price, and a factor
that is not perfectly mobile adds its price in each activity that uses it. A
finite-difference Jacobian costs one residual evaluation per unknown, which is
two or three evaluations in the standard model and several hundred with a curve
or a sluggish factor in every sector. Here every step of the evaluation carries
a tangent with one row per unknown:

- prices follow Shephard's lemma, so the value-added, Armington and CET prices
  differentiate into the intensities the solver already computes;
- the purchaser-price fixed point is linear in its tangent, which is solved
  directly rather than iterated;
- absorption solves ``(I - B G - mq) A = F``, so its tangent is one more solve
  with that matrix, one right-hand side per unknown;
- the government budget and saving, which the solver finds in closed form from
  probes, are differentiated implicitly: their tangent is the one at which the
  budget gaps stay at zero.

:func:`unsupported` is the one place that decides coverage. It is a list of what
the derivation reads, not of what it leaves out: an option that is not on it,
including one added to the model after this module was written, keeps the
finite-difference Jacobian until its algebra is derived here and its name added.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .functions import armington, cet

__all__: list[str] = []


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
# Each set names the fields the derivation covers at any value they can take.
# A field in a ``_VALUES`` table is covered at the values listed. Every other
# field must be at its declared default, which is how an option is off.

_SPEC = frozenset({
    # Account names and layout.
    "activity_prefix", "commodity_prefix", "household_prefix", "factors", "government",
    "investment", "rest_of_world", "activities", "commodities", "households",
    "tariff_account", "indirect_tax_account", "export_tax_account", "numeraire_factor",
    "margins", "negative_value_added",
    # Elasticities of the blocks derived here.
    "sigma_va", "sigma_armington", "sigma_cet", "frisch", "export_demand", "import_supply",
    "factor_mobility", "factor_mobility_form",
    # Parameters of blocks that are off unless the field that switches them on is set, and
    # that field is not on this list.
    "sigma_vae", "sigma_energy", "energy_nest", "energy_capital", "sigma_ke",
    "leisure_factor", "sigma_interregional", "vintage_sigma", "vintage_substitution",
    "fixed_cost_factor",
})
_SPEC_VALUES = {
    "value_added": ("ces", "cobb_douglas", "leontief"),
    "export_rule": ("fixed_shares", "cet"),
    "household_demand": ("cobb_douglas", "les"),
}
_ECONOMY = frozenset({
    "layout", "spec", "X0", "Q0", "io", "msh", "theta", "vunit", "aunit", "leontief",
    "own_account", "sigma_va", "sigma_arm", "ptax", "atax", "imp0", "exp0", "dom0",
    "exp_share", "dom_share", "A0", "mq", "arm_share", "hh_budget", "hh_savrate",
    "hh_facshare", "hh_direct", "hh_committed", "hh_income0", "pop_share", "gov_dem",
    "inv_dem", "fac_supply", "trade_balance0", "foreign_sav", "diagnostics", "sigma_cet",
    "hh_dtax", "hh_gov_transfer", "hh_remit_net", "fac_gov_share", "fac_row_share",
    "hh_spend0", "hh_gross0", "tm0", "tauz0", "te0", "gov_savrate", "gov_saving0",
    "gov_revenue0", "mobility",
    # Read only by reports, or only once a scenario prices carbon.
    "emissions", "physical_units",
})
_CLOSURE = frozenset({
    "numeraire", "tolerance", "exchange_rate", "labour_factor", "labour_supply_elasticity",
    # Read together with ``government_target``, which is checked below.
    "government_target_value", "government_adjusts",
    # Read only by the labour closures that are not on the list below.
    "real_wage", "wage_curve_elasticity", "natural_unemployment",
})
_CLOSURE_VALUES = {
    "investment": ("savings_driven", "fixed_real"),
    "government": ("real_fixed", "nominal_fixed", "saving_rate"),
    "foreign": ("fixed_saving", "fixed_rate"),
    "foreign_saving_unit": ("numeraire", "foreign_currency"),
    "labour": ("full_employment", "elastic_supply"),
    "government_target": (None, "saving"),
}
_SCENARIO = frozenset({
    "name", "factor_supply", "pool_supply", "factor_ownership", "productivity",
    "activity_scale", "product_tax", "product_tax_firm", "transfers", "government_absorbs",
    "world_price", "tariff", "output_tax", "export_tax", "export_price", "investment_level",
    "investment_mix", "foreign_factor_share", "fdi",
    # Read only with a wedge, a capacity limit or a quota, none of which is on this list.
    "wedge_revenue_to", "wedge_revenue_factor", "capacity_owner", "quota_rent_to",
})
# A wedge or a carbon price of zero is the same model as none.
_OFF_AT_ZERO = frozenset({"consumer_wedge", "government_wedge", "carbon_price"})


def _is_default(value, spec_field) -> bool:
    if spec_field.default is not dataclasses.MISSING:
        default = spec_field.default
    elif spec_field.default_factory is not dataclasses.MISSING:
        default = spec_field.default_factory()
    else:
        return False                       # a required field has no value that means "off"
    if value is None or default is None:
        return value is default
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return False
    try:
        return bool(value == default)
    except (TypeError, ValueError):
        return False


def unsupported(econ, closure, scenario) -> str | None:
    """The first option outside the derivation, by name, or ``None`` when all are covered.

    The solver takes the exact Jacobian only on ``None``. Anything else is the
    reason it differences the residual numerically instead.
    """
    for obj, label, known, values in ((econ.spec, "ModelSpec", _SPEC, _SPEC_VALUES),
                                      (econ, "Economy", _ECONOMY, {}),
                                      (closure, "Closure", _CLOSURE, _CLOSURE_VALUES),
                                      (scenario, "Scenario", _SCENARIO, {})):
        for spec_field in dataclasses.fields(obj):
            name = spec_field.name
            value = getattr(obj, name)
            if name in values:
                if value not in values[name]:
                    return f"{label}.{name}={value!r}"
            elif name in known or _is_default(value, spec_field):
                continue
            elif name in _OFF_AT_ZERO and not np.any(np.asarray(value, dtype=float) != 0):
                continue
            else:
                return f"{label}.{name}"
    if closure.government_target is not None and closure.government_adjusts != "spending":
        return f"Closure.government_adjusts={closure.government_adjusts!r}"
    for mob in econ.mobility.values():
        if mob.form not in ("cet", "logit") or mob.allow_new:
            return f"ModelSpec.factor_mobility_form of {mob.factor!r}"
    return None


# ---------------------------------------------------------------------------
# The Jacobian
# ---------------------------------------------------------------------------

def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den != 0)


def _pool_tangent(mob, wage, dwage, supply):
    """Tangent of the supply of a sluggish factor to each activity, summed over its pools.

    ``supply`` is what :meth:`FactorMobility.supply` returned at ``wage``. Each
    pool moves towards the activities whose price rose against its own index:
    the index weights are value shares under the CET and head-count shares under
    the logit.
    """
    users = mob.users
    rel = np.zeros_like(dwage)
    rel[:, users] = dwage[:, users] / wage[users]
    weight = supply * wage[None, :] if mob.form == "cet" else supply
    weight = _safe_div(weight, weight.sum(axis=1, keepdims=True))
    index = rel @ weight.T                                       # (n, n_pool)
    return mob.omega * (supply.sum(axis=0)[None, :] * rel - index @ supply)


def jacobian(econ, ctx, state) -> np.ndarray:
    """Residual Jacobian at the point whose blocks are in ``state``.

    ``ctx`` holds what ``solve_period`` fixed for the period: the shock, the
    closure and the layout of the unknowns. ``state["at"]`` is the point, and
    the rest of ``state`` what the residual computed there. Rows follow the
    residual vector, columns the unknowns.
    """
    from .solve import _laspeyres_purchaser, factor_intensity, value_added_price

    closure = ctx["closure"]
    nf, nc, na = econ.n_fac, econ.n_com, econ.n_act
    n = ctx["n_unknown"]
    z = np.atleast_1d(np.asarray(state["at"], dtype=float))
    nu, others, mobility = ctx["nu"], ctx["others"], econ.mobility
    i_ext, i_inv, i_curve, i_mob = ctx["i_ext"], ctx["i_inv"], ctx["i_curve"], ctx["i_mob"]
    curve_x, curve_m = ctx["curve_x"], ctx["curve_m"]
    supply = ctx["supply"]
    fixed_inv = closure.investment == "fixed_real"

    # ------------------------------------------------------------------
    # The unknowns and their tangents: log prices, the external unknown, the
    # levy, log world prices on a curve, log prices of a sluggish factor.
    # ------------------------------------------------------------------
    w = np.full(nf, nu)
    dw = np.zeros((n, nf))
    for k, i in enumerate(others):
        w[i] = nu * np.exp(z[k])
        dw[k, i] = w[i]
    Wm = np.tile(w[:, None], (1, na))
    dWm = np.repeat(dw[:, :, None], na, axis=2)
    for i, mob in mobility.items():
        Wm[i, mob.users] = nu * np.exp(z[i_mob[i]])
        dWm[i_mob[i], i, np.where(mob.users)[0]] = Wm[i, mob.users]
    der = np.zeros(n)
    dforeign_sav = np.zeros(n)
    if closure.solves_exchange_rate:
        er = float(np.exp(z[i_ext]))
        der[i_ext] = er
        if ctx["in_dollars"]:
            dforeign_sav = der * ctx["trade_target"]
    else:
        er = closure.exchange_rate
        dforeign_sav[i_ext] = ctx["ext_scale"]
    dlevy = np.zeros(n)
    if fixed_inv:
        dlevy[i_inv] = ctx["sav_scale"]
    k_x = i_curve + np.arange(curve_x.size)
    k_m = i_curve + curve_x.size + np.arange(curve_m.size)
    world_pe = ctx["world_pe"].copy()
    world_pe[curve_x] *= np.exp(z[k_x])
    world_p = ctx["world_p"].copy()
    world_p[curve_m] *= np.exp(z[k_m])
    dworld_pe = np.zeros((n, nc))
    dworld_pe[k_x, curve_x] = world_pe[curve_x]
    dworld_p = np.zeros((n, nc))
    dworld_p[k_m, curve_m] = world_p[curve_m]

    tfp = ctx["tfp"]
    io = econ.io / tfp[None, :]
    vunit = econ.vunit / tfp
    ptax_a, ptax_f = ctx["ptax_a"], ctx["ptax_f"]
    tm, tauz, te = ctx["tm"], ctx["tauz"], ctx["te"]
    tf0, tz0, te_base, ex_wedge = ctx["tf0"], ctx["tz0"], ctx["te_base"], ctx["ex_wedge"]
    cet_on, sigma_cet = ctx["cet_on"], ctx["sigma_cet"]

    # ------------------------------------------------------------------
    # Prices at the converged purchaser prices.
    # ------------------------------------------------------------------
    pva = value_added_price(econ, Wm)
    fi = factor_intensity(econ, Wm, vunit=vunit, aunit=econ.aunit / tfp[None, :])
    dpva = np.einsum("nfa,fa->na", dWm, fi)                  # Shephard: dpva = fi' dw
    pm_border = er * world_p
    dpm_border = der[:, None] * world_p[None, :] + er * dworld_p
    to_buyer = (np.ones(nc) if tm is None else 1.0 + tm) / tf0
    pm = pm_border * to_buyer
    dpm = dpm_border * to_buyer[None, :]
    pe_fob = er * world_pe
    dpe_fob = der[:, None] * world_pe[None, :] + er * dworld_pe
    pe = pe_fob * ex_wedge
    dpe = dpe_fob * ex_wedge[None, :]
    out_wedge = np.ones(nc) if tauz is None else (1.0 + tauz) / tz0
    pqf, pq, pbase, pd = state["pqf"], state["pq"], state["pbase"], state["pd"]
    p_dom = state["p_dom"]
    p0 = pd * out_wedge
    arm, dsh, msh = armington(econ.dom_share, p_dom, pm, econ.sigma_arm)
    if cet_on:
        # p_dom solves pz(p_dom, pe) = p0, and pz differentiates into the CET's
        # own fractions, so dp_dom = (dp0 - exp_frac dpe) / dom_frac. A commodity
        # that is sold whole to one market takes the output price.
        pz, dom_frac, exp_frac = cet(econ.exp_share, p_dom, pe, sigma_cet)
        interior = (econ.exp_share > 0) & (econ.exp_share < 1)
        from_p0 = np.where(interior, _safe_div(1.0, dom_frac), 1.0)
        from_pe = np.where(interior, -_safe_div(exp_frac, dom_frac), 0.0)
    else:
        dom_frac, exp_frac = 1.0 - econ.exp_share, econ.exp_share
        from_p0, from_pe = np.ones(nc), np.zeros(nc)
    home = dsh * econ.arm_share                              # domestic variety per unit absorbed
    abroad = msh * econ.arm_share

    # The fixed point pqf = (1 + ptax_f) pbase(pqf) is linear in its tangent:
    # dpqf = dpqf T + c, with T the pass-through of purchaser prices into
    # themselves by way of intermediate costs and margins.
    through = from_p0 * out_wedge * home
    T = ((io @ econ.msh) * through[None, :] + econ.mq) * (1.0 + ptax_f)[None, :]
    c = (((dpva * vunit[None, :]) @ econ.msh) * through[None, :]
         + dpe * (from_pe * home)[None, :] + dpm * abroad[None, :]) * (1.0 + ptax_f)[None, :]
    dpqf = np.linalg.solve((np.eye(nc) - T).T, c.T).T
    dpa = dpqf @ io + dpva * vunit[None, :]
    dpd = dpa @ econ.msh
    dp0 = dpd * out_wedge[None, :]
    dp_dom = dp0 * from_p0[None, :] + dpe * from_pe[None, :]
    darm = dsh[None, :] * dp_dom + msh[None, :] * dpm
    dpbase = darm * econ.arm_share[None, :] + dpqf @ econ.mq
    dpq = dpbase * (1.0 + ptax_a)[None, :]
    sigma_arm = np.broadcast_to(np.asarray(econ.sigma_arm, dtype=float), (nc,))
    ddsh = sigma_arm[None, :] * dsh[None, :] * (_safe_div(darm, arm[None, :])
                                                - _safe_div(dp_dom, p_dom[None, :]))
    dmsh = sigma_arm[None, :] * msh[None, :] * (_safe_div(darm, arm[None, :])
                                                - _safe_div(dpm, pm[None, :]))
    if cet_on:
        dpz = dom_frac[None, :] * dp_dom + exp_frac[None, :] * dpe
        s = np.broadcast_to(np.asarray(sigma_cet, dtype=float), (nc,))[None, :]
        ddom_frac = s * dom_frac[None, :] * (_safe_div(dp_dom, p_dom[None, :])
                                             - _safe_div(dpz, pz[None, :]))
        dexp_frac = s * exp_frac[None, :] * (_safe_div(dpe, pe[None, :])
                                             - _safe_div(dpz, pz[None, :]))
        p_exp, dp_exp = pe_fob, dpe_fob
    else:
        ddom_frac = np.zeros((n, nc))
        dexp_frac = np.zeros((n, nc))
        p_exp = p0 / ex_wedge
        dp_exp = dp0 / ex_wedge[None, :]

    # ------------------------------------------------------------------
    # Factor income, household budgets and demand.
    # ------------------------------------------------------------------
    lab = ctx["lab"]
    paid = supply.astype(float).copy()
    dpaid = np.zeros((n, nf))
    if closure.labour == "elastic_supply":
        # The index is linear in the prices, so its weights are its values at
        # the unit vectors.
        if "price_index" not in ctx:
            ctx["price_index"] = np.array([_laspeyres_purchaser(econ, unit)
                                           for unit in np.eye(nc)])
        index = ctx["price_index"]
        P = float(pq @ index)
        eps = closure.labour_supply_elasticity
        paid[lab] = supply[lab] * (w[lab] / P) ** eps
        dpaid[:, lab] = paid[lab] * eps * (dw[:, lab] / w[lab] - (dpq @ index) / P)
    fac_inc = w * paid
    dfac_inc = dw * paid[None, :] + w[None, :] * dpaid
    pool_supply, dpool_supply = {}, {}
    for i, mob in mobility.items():
        S, _ = mob.supply(Wm[i], ctx["endow"][i])
        pool_supply[i] = S.sum(axis=0)
        dpool_supply[i] = _pool_tangent(mob, Wm[i], dWm[:, i, :], S)
        # Pool income is the activity prices times the supplies, in both forms.
        dfac_inc[:, i] = dpool_supply[i] @ Wm[i] + dWm[:, i, :] @ pool_supply[i]
    fac_gov, fac_row = ctx["fac_gov"], ctx["fac_row"]
    dbase_income = ((dfac_inc * (1.0 - fac_gov - fac_row)[None, :]) @ ctx["facshare"].T
                    - dlevy[:, None] * econ.pop_share[None, :])
    dtax = ctx["dtax"]
    net = (1.0 - dtax) * (1.0 - econ.hh_savrate)
    dhh_income = dbase_income                                # no wedge revenue to recycle
    ddirect = (dpa * ctx["act_scale"][None, :]) @ econ.hh_direct.T
    dcom_spend = dhh_income * net[None, :] - ddirect
    # Demand is base + share * budget / price under both systems: the
    # supernumerary budget and the marginal shares of the LES, or the whole
    # budget and the Cobb-Douglas shares.
    share = ctx["beta_les"]
    if econ.spec.household_demand == "les":
        budget = state["hh_com_spend"] - pq @ econ.hh_committed.T
        dbudget = dcom_spend - dpq @ econ.hh_committed.T
    else:
        budget, dbudget = state["hh_com_spend"], dcom_spend
    dhh_final = (dbudget @ share) / pq[None, :] - (budget @ share)[None, :] * dpq / pq[None, :] ** 2
    hh_final = state["hh_q"].sum(axis=0)
    dhh_dtax = dhh_income * dtax[None, :]
    dhh_sav = ((dhh_income - dhh_dtax) * econ.hh_savrate[None, :]).sum(axis=1)
    dgov_inst = dhh_dtax.sum(axis=1) + dfac_inc @ fac_gov
    drow_fac = dfac_inc @ fac_row

    # ------------------------------------------------------------------
    # Absorption: the tangent of the linear system, not of its inverse.
    # ------------------------------------------------------------------
    A, Q, X = state["A"], state["Q"], state["X"]
    gov_q, inv_q = state["gov_q"], state["inv_q"]
    den = dom_frac * tz0
    floored = den <= 1e-12
    den = np.maximum(den, 1e-12)
    gp = home / den                                          # output per unit absorbed
    dgp = (ddsh * econ.arm_share[None, :]
           - gp[None, :] * np.where(floored, 0.0, tz0)[None, :] * ddom_frac) / den[None, :]
    Bm = io @ econ.msh
    system = lu_factor(np.eye(nc) - Bm * gp[None, :] - econ.mq)
    moved = (dgp * A[None, :]) @ Bm.T                        # B dG A
    fin_c = hh_final + gov_q
    atax_by_com = econ.msh.T @ (econ.atax * nu)
    gov_free = closure.government == "saving_rate" or (
        closure.government_target == "saving" and closure.government_adjusts == "spending")
    sav_free = closure.investment == "savings_driven"
    gov_budget = float(pq @ gov_q)
    saving = float(state["savings"])

    def flows(dg, ds):
        """Tangent of the circular flow, the two free budgets moving by ``dg`` and ``ds``."""
        if gov_free:
            dgov_q = ctx["gov_w"][None, :] * (dg[:, None] / pq[None, :]
                                              - gov_budget * dpq / pq[None, :] ** 2)
        elif closure.government == "nominal_fixed":
            dgov_q = -gov_q[None, :] * dpq / pq[None, :]
        else:
            dgov_q = np.zeros((n, nc))
        if sav_free:
            dinv_q = ctx["inv_share"][None, :] * (ds[:, None] / pqf[None, :]
                                                  - saving * dpqf / pqf[None, :] ** 2)
        else:
            dinv_q = np.zeros((n, nc))
        dfin = dhh_final + dgov_q
        dA = lu_solve(system, (dfin + dinv_q + moved).T).T
        dQ = dgp * A[None, :] + gp[None, :] * dA
        revenue = ((dpbase * ((A - fin_c) * ptax_f + fin_c * ptax_a)[None, :]
                    + pbase[None, :] * ((dA - dfin) * ptax_f[None, :] + dfin * ptax_a[None, :])
                    ).sum(axis=1) + dQ @ atax_by_com)
        if tm is not None:
            revenue += ((tm * econ.arm_share / tf0)[None, :]
                        * (dpm_border * (msh * A)[None, :] + pm_border[None, :] * dmsh * A[None, :]
                           + (pm_border * msh)[None, :] * dA)).sum(axis=1)
        if tauz is not None:
            revenue += (tauz[None, :] * (dpd * Q[None, :] + pd[None, :] * dQ)).sum(axis=1)
        if te is not None:
            border = exp_frac * tz0 * Q / (1.0 - te_base)
            dborder = ((dexp_frac * Q[None, :] + exp_frac[None, :] * dQ)
                       * (tz0 / (1.0 - te_base))[None, :])
            revenue += (te[None, :] * (dp_exp * border[None, :]
                                       + p_exp[None, :] * dborder)).sum(axis=1)
        dgov_income = revenue + dgov_inst
        dgov_spend = (dpq * gov_q[None, :] + pq[None, :] * dgov_q).sum(axis=1)
        davailable = dhh_sav + dlevy + dgov_income - dgov_spend + dforeign_sav
        return dA, dQ, dgov_income, davailable

    def gaps(flow, dg, ds):
        """Tangent of the budget conditions the solver holds at zero."""
        _, _, dgov_income, davailable = flow
        out = []
        if gov_free:
            kept = (1.0 - ctx["gov_savrate"]) if closure.government == "saving_rate" else 1.0
            out.append(dg - kept * dgov_income)
        if sav_free:
            out.append(ds - davailable)
        return np.array(out)

    # The solver holds the budget conditions at zero at every point, so the
    # budgets move with the unknowns by exactly what keeps them there. The
    # conditions are affine in the budgets: one probe each gives their slope.
    zero, one = np.zeros(n), np.ones(n)
    free = [name for name, on in (("gov", gov_free), ("sav", sav_free)) if on]
    flow = flows(zero, zero)
    if free:
        held = gaps(flow, zero, zero)                        # (n_free, n)
        slope = np.empty((len(free), len(free)))
        for k, name in enumerate(free):
            dg, ds = (one, zero) if name == "gov" else (zero, one)
            slope[:, k] = (gaps(flows(dg, ds), dg, ds) - held)[:, 0]
        moves = np.linalg.solve(slope, -held)
        dg = moves[free.index("gov")] if gov_free else zero
        ds = moves[free.index("sav")] if sav_free else zero
        flow = flows(dg, ds)
    dA, dQ, _, davailable = flow

    # ------------------------------------------------------------------
    # Factor markets, the current account and the curves.
    # ------------------------------------------------------------------
    dX = dQ @ econ.msh.T
    sig_va = (np.ones(na) if econ.spec.value_added == "cobb_douglas"
              else np.broadcast_to(np.asarray(econ.sigma_va, dtype=float), (na,)))
    sig_va = np.where(econ.leontief, 0.0, sig_va)
    dfi = sig_va[None, None, :] * fi[None, :, :] * (
        _safe_div(dpva, pva[None, :])[:, None, :] - _safe_div(dWm, Wm[None, :, :]))
    fac_per = fi * vunit[None, :]
    dfac_by_act = (dfi * (vunit * X)[None, None, :]
                   + fac_per[None, :, :] * dX[:, None, :])    # (n, n_fac, n_act)
    fac_dem = fac_per @ X
    dfac_dem = dfac_by_act.sum(axis=2)
    reference = supply.astype(float).copy()
    dreference = np.zeros((n, nf))
    if closure.labour == "elastic_supply":
        reference[lab] = paid[lab]
        dreference[:, lab] = dpaid[:, lab]
    # The gap is over the larger of the reference and one, and under an elastic
    # supply the reference moves: the quotient rule, where it is the larger.
    scale = np.maximum(np.abs(reference), 1.0)
    dscale = np.where(np.abs(reference) > 1.0, np.sign(reference), 0.0)[None, :] * dreference
    dgaps = ((dfac_dem - dreference) / scale[None, :]
             - ((fac_dem - reference) / scale ** 2)[None, :] * dscale)

    imports = abroad * A / tf0
    dimports = (dmsh * A[None, :] + msh[None, :] * dA) * (econ.arm_share / tf0)[None, :]
    exports = exp_frac * tz0 * Q / (1.0 - te_base)
    dexports = ((dexp_frac * Q[None, :] + exp_frac[None, :] * dQ)
                * (tz0 / (1.0 - te_base))[None, :])
    dcurrent = ((dpm_border * imports[None, :] + pm_border[None, :] * dimports).sum(axis=1)
                - (dp_exp * exports[None, :] + p_exp[None, :] * dexports).sum(axis=1)
                + drow_fac)

    rows = [dgaps[:, i] for i in others]
    rows.append((dcurrent - dforeign_sav) / ctx["ext_scale"])
    if fixed_inv:
        rows.append((davailable - dpqf @ inv_q) / ctx["sav_scale"])
    for j, i in enumerate(curve_x):
        row = _safe_div(dexports[:, i], exports[i])
        row[k_x[j]] += ctx["eta_x"][i]
        rows.append(row)
    for j, i in enumerate(curve_m):
        row = _safe_div(dimports[:, i], imports[i])
        row[k_m[j]] -= ctx["psi_m"][i]
        rows.append(row)
    for i, mob in mobility.items():
        # A sluggish factor clears activity by activity, against what its pools
        # supply there.
        users = np.where(mob.users)[0]
        gap = (dfac_by_act[:, i, :] - dpool_supply[i]) / np.maximum(
            np.abs(ctx["base_use"][i]), 1.0)[None, :]
        rows.extend(gap[:, a] for a in users)
    out = np.asarray(rows, dtype=float).reshape(len(rows), n)
    if out.shape != (n, n):
        raise RuntimeError(
            f"the exact national Jacobian has {out.shape[0]} rows for {n} unknowns: the "
            f"residual has a block that cgelab.national_jacobian.unsupported did not decline")
    return out
