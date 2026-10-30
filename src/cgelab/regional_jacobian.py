"""The exact Jacobian of the regional residual, by forward-mode differentiation.

The regional solver's unknowns are the factor prices, the exchange rate and,
under fixed real investment, the levy that finances it. Each residual evaluation
solves a price fixed point and a linear system for absorption, so a
finite-difference Jacobian costs one full evaluation per unknown. Here every
step of that evaluation carries a tangent with one column per unknown:

- prices follow Shephard's lemma, so each CES, Armington and value-added price
  differentiates into the intensities the solver already computes;
- the price fixed point is differentiated at its solution, by iterating the
  tangent map, which contracts at the rate the prices themselves did;
- absorption solves ``(I - B - v t') A = F`` (``v t'`` only when investment
  takes the saving that taxes provide), so its tangent is one more solve with
  that matrix, one right-hand side per unknown.

The result is exact to rounding, where finite differences carry an error of
the order of their step. :func:`supported` names the closures it covers; the
others (a government that spends a share of its revenue, and targets met by a
tax rate) keep the finite-difference Jacobian.
"""

from __future__ import annotations

import numpy as np

from .functions import armington, ces_intensity, ces_price, cet, cet_domestic_price
from .solve import factor_intensity, value_added_price

__all__: list[str] = []


def supported(closure) -> bool:
    """Whether :func:`jacobian` covers this closure."""
    dynamic = (closure.government == "saving_rate"
               or (closure.government_target == "saving"
                   and closure.government_adjusts == "spending"))
    lever = (closure.government_target is not None
             and closure.government_adjusts in ("direct_tax", "product_tax"))
    return not dynamic and not lever


def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den != 0)


def jacobian(econ, ctx, prices, qty) -> np.ndarray:
    """Residual Jacobian at the point where ``prices`` and ``qty`` were computed.

    ``ctx`` holds what ``solve_regions`` fixed for the period: the factor and
    exchange-rate values at the point, the shock, the closure and the layout of
    the unknowns. Rows follow the residual vector, columns the unknowns.
    """
    reg = econ.regional
    R, K, C = reg.n_reg, reg.n_dest, reg.n_local
    nf, na = econ.n_fac, econ.n_act
    n = ctx["n_unknown"]
    w, er, nu = ctx["w"], ctx["er"], ctx["nu"]
    tfp, supply = ctx["tfp"], ctx["supply"]
    fixed_inv = ctx["fixed_investment"]

    # Tangents of the unknowns: log prices, the log exchange rate, the levy.
    dw = np.zeros((n, nf))
    for k, i in enumerate(ctx["others"]):
        dw[k, i] = w[i]
    der = np.zeros(n)
    der[ctx["i_er"]] = er
    dlevy = np.zeros(n)
    if fixed_inv:
        dlevy[ctx["i_inv"]] = 1.0

    # ------------------------------------------------------------------
    # Prices at the converged purchaser prices.
    # ------------------------------------------------------------------
    pq = prices["pq"]
    ptax = ctx["ptax"]
    yard = 1.0 + ptax
    io = reg.io / tfp[None, :]
    vunit = econ.vunit / tfp
    pva = value_added_price(econ, w)
    fi = factor_intensity(econ, w, vunit=vunit, aunit=econ.aunit / tfp[None, :])
    dpva = dw @ fi                                   # Shephard: dpva = fi' dw
    pe = er * ctx["export_world"]
    pm = er * ctx["world"]
    dpe = der[:, None] * ctx["export_world"][None, :]
    dpm = der[:, None] * ctx["world"][None, :]
    pa = (io.T * pq[reg.activity_region]).sum(axis=1) + pva * vunit + econ.atax * nu
    p0 = (econ.msh.T @ pa)[reg.origin_index]
    if reg.cet:
        p_rc = cet_domestic_price(reg.exp_share, p0, pe[None, :], reg.sigma_cet)
    else:
        p_rc = p0
    if reg.transport is None:
        coef = np.zeros((R, K, C))
        pt = 0.0
        pool = np.zeros((R, C))
    else:
        coef = reg.transport.coefficient
        pt = reg.transport.pool_price(p_rc)
        pool = reg.transport.supply / reg.transport.supply.sum()
    p_del = p_rc[:, None, :] + coef * pt
    shares = reg.destination_share.reshape(R, K * C)
    sigma_col = np.tile(reg.sigma, K)
    pd = ces_price(shares, p_del.reshape(R, K * C), sigma_col).reshape(K, C)
    phi = ces_intensity(shares, p_del.reshape(R, K * C), pd.ravel(),
                        sigma_col).reshape(R, K, C)
    sigma_arm = np.broadcast_to(reg.sigma_arm, (K, C))
    arm_p, dsh, msh = armington(reg.dom_share_dest, pd, np.broadcast_to(pm, (K, C)), sigma_arm)

    def cet_tangent(dp0):
        if not reg.cet:
            return dp0
        e = 1.0 + reg.sigma_cet
        se = reg.exp_share
        sd = 1.0 - se
        interior = (se > 0) & (sd > 0)
        num = p0 ** (e - 1.0) * dp0 - se * pe[None, :] ** (e - 1.0) * dpe[:, None, :]
        return np.where(interior[None], _safe_div(num, (sd * p_rc ** (e - 1.0))[None]), dp0)

    def through_prices(dpq):
        dpa = np.einsum("cj,njc->nj", io, dpq[:, reg.activity_region, :]) + dpva * vunit
        dp0 = (dpa @ econ.msh)[:, reg.origin_index]
        dprc = cet_tangent(dp0)
        if reg.transport is None:
            dpt = np.zeros(n)
        else:
            dpt = pt * (pool[None] * _safe_div(dprc, p_rc[None])).sum(axis=(1, 2))
        dpdel = dprc[:, :, None, :] + coef[None] * dpt[:, None, None, None]
        dpd = (phi[None] * dpdel).sum(axis=1)
        darm = dsh[None] * dpd + msh[None] * dpm[:, None, :]
        return dpa, dp0, dprc, dpt, dpdel, dpd, darm

    # The tangent of the fixed point pq = yard (arm_share arm(pq) + pq mq).
    dpq = np.zeros((n, K, C))
    scale = max(float(np.max(np.abs(dw))), abs(er), 1.0)
    for _ in range(1000):
        *_, darm = through_prices(dpq)
        new = (darm * reg.arm_share + dpq @ reg.mq) * yard
        done = np.max(np.abs(new - dpq)) <= 1e-15 * scale
        dpq = new
        if done:
            break
    else:
        raise RuntimeError("the tangent of the regional price fixed point did not converge")
    dpa, dp0, dprc, dpt, dpdel, dpd, darm = through_prices(dpq)
    sig = reg.sigma[None, None, None, :]
    dphi = sig * phi[None] * (_safe_div(dpd, pd)[:, None] - _safe_div(dpdel, p_del[None]))
    ddsh = sigma_arm[None] * dsh[None] * (_safe_div(darm, arm_p) - _safe_div(dpd, pd))
    dmsh = sigma_arm[None] * msh[None] * (_safe_div(darm, arm_p)
                                          - _safe_div(dpm[:, None, :], pm[None, None, :]))
    pbase = pq / yard
    dpbase = dpq / yard
    if reg.cet:
        _, dom_frac, exp_frac = cet(reg.exp_share, p_rc, pe[None, :], reg.sigma_cet)
        s = reg.sigma_cet[None]
        ddf = s * dom_frac[None] * (_safe_div(dprc, p_rc[None]) - _safe_div(dp0, p0[None]))
        def_ = s * exp_frac[None] * (_safe_div(dpe[:, None, :], pe[None, None, :])
                                     - _safe_div(dp0, p0[None]))
    else:
        dom_frac, exp_frac = 1.0 - reg.exp_share, reg.exp_share
        ddf = def_ = np.zeros((n, R, C))

    # ------------------------------------------------------------------
    # Incomes and final demand.
    # ------------------------------------------------------------------
    fac_inc = w * supply
    dfac_inc = dw * supply[None]
    migrated = {}
    for local, item in reg.migration.items():
        idx = item["indices"]
        res = qty["migration"][local]
        S, W, L = res["supply"], res["price"], supply[idx]
        wl, dlog = w[idx], dw[:, idx] / w[idx][None]
        om = item["model"].omega
        delta = np.maximum(item["model"].delta, 1e-300)
        if item["model"].form == "cet":
            value = delta * wl[None, :] ** (1.0 + om) / W[:, None] ** (1.0 + om)
            dlogW = (value[None] * dlog[:, None, :]).sum(axis=2)
            dS = om * S[None] * (dlog[:, None, :] - dlogW[:, :, None])
            dW = W[None] * dlogW
        else:
            share = S / np.maximum(L[:, None], 1e-300)
            mean = (share[None] * dlog[:, None, :]).sum(axis=2, keepdims=True)
            dshare = om * share[None] * (dlog[:, None, :] - mean)
            dS = dshare * L[None, :, None]
            dW = (dshare * wl[None, None, :]).sum(axis=2) + (share[None] * dw[:, None, idx]).sum(axis=2)
        fac_inc[idx] = W * L
        dfac_inc[:, idx] = dW * L[None]
        migrated[local] = (idx, dS)
    fac_gov, fac_row = ctx["fac_gov"], ctx["fac_row"]
    fgr = reg.fac_gov_regional.sum(axis=0)
    dkept = dfac_inc * (1.0 - fac_gov - fac_row - fgr)[None]
    dhh = dkept @ ctx["facshare"].T - dlevy[:, None] * econ.pop_share[None]
    dtax = ctx["dtax"]
    hh_income = qty["hh_income"]
    ddisp = dhh * (1.0 - dtax)[None]
    dsav = ddisp * econ.hh_savrate[None]
    dspend = ddisp - dsav
    spend = qty["spend"]
    hreg = reg.household_region
    pqh = pq[hreg]
    dhhq = reg.hh_budget[None] * (dspend[:, :, None] / pqh[None]
                                  - spend[None, :, None] * dpq[:, hreg, :] / pqh[None] ** 2)
    owner = np.zeros((K, econ.n_hh))
    owner[hreg, np.arange(econ.n_hh)] = 1.0
    dhh_by = np.einsum("kh,nhc->nkc", owner, dhhq)
    gq = qty["gov_channel_q"]
    if ctx["closure"].government == "nominal_fixed":
        dgq = -gq[None] * dpq / pq[None]
    else:
        dgq = np.zeros((n, K, C))
    dgov_spend = (dpq * gq[None] + pq[None] * dgq).sum(axis=(1, 2))
    dincome_excl = (dfac_inc @ (fac_gov + fgr)) + dhh @ dtax
    gov_spend = float((pq * gq).sum())
    income_excl = (float((fac_inc * fac_gov).sum()) + float((fac_inc * fgr).sum())
                   + float((hh_income * dtax).sum()) - float(ctx["gov_tr"].sum()))
    available0 = (float(qty["hh_sav"].sum()) + income_excl - gov_spend
                  + ctx["foreign_target"] + ctx["levy"])
    davail0 = dsav.sum(axis=1) + dincome_excl - dgov_spend + dlevy

    inv_channel = np.zeros((K, C))
    inv_channel[-1] = reg.inv_dem
    inv_channel[:R] = reg.inv_dem_regional
    inv_total = inv_channel.sum(axis=0)
    inv_base_q = inv_channel / np.maximum(reg.base_price, 1e-12)
    inv_base_cost = float(inv_total.sum())
    inv_price = np.divide((inv_base_q * pq).sum(axis=0), inv_total,
                          out=np.ones(C), where=inv_total > 0)
    dinv_price = np.where(inv_total > 0, (inv_base_q[None] * dpq).sum(axis=1)
                          / np.where(inv_total > 0, inv_total, 1.0), 0.0)
    v = (inv_base_q / inv_base_cost if inv_base_cost > 0 else inv_base_q) / np.maximum(
        inv_price[None, :], 1e-12)
    dv = -v[None] * dinv_price[:, None, :] / np.maximum(inv_price, 1e-12)[None, None, :]

    # ------------------------------------------------------------------
    # The absorption system.
    # ------------------------------------------------------------------
    arm = reg.arm_share
    link = phi * (dsh * arm)[None]
    dlink = dphi * (dsh * arm)[None, None] + phi[None] * (ddsh * arm)[:, None]
    if reg.transport is None:
        ss = np.zeros((R, C))
        dss = np.zeros((n, R, C))
    else:
        ss = _safe_div(pool * pt, p_rc)
        dss = ss[None] * (_safe_div(dpt, pt)[:, None, None] - _safe_div(dprc, p_rc[None]))
    tl = (coef * link).sum(axis=0)
    dtl = (coef[None] * dlink).sum(axis=1)
    dfr = np.maximum(dom_frac, 1e-12)
    ol = (link + ss[:, None, :] * tl[None]) / dfr[:, None, :]
    dol = ((dlink + dss[:, :, None, :] * tl[None, None] + ss[None, :, None, :] * dtl[:, None])
           / dfr[None, :, None, :] - ol[None] * ddf[:, :, None, :] / dfr[None, :, None, :])
    msh_rc = econ.msh[:, reg.origin_index]
    M = np.einsum("arc,rdc->adc", msh_rc, ol).reshape(na, K * C)
    IO = np.zeros((K, C, na))
    IO[reg.activity_region, :, np.arange(na)] = io.T
    IO = IO.reshape(K * C, na)
    B = IO @ M + np.kron(np.eye(K), reg.mq)
    atax = econ.atax * nu
    t = (pbase * ptax).ravel() + M.T @ atax
    A = qty["A_dest"].ravel()
    # dB A and dM' atax without forming dM (n x activities x destinations x
    # goods) or dB (n x (K C)^2): both contract the output links first.
    dMA = ((dol * A.reshape(K, C)[None, None]).sum(axis=2).reshape(n, R * C)
           @ msh_rc.reshape(na, R * C).T)
    taxed = np.einsum("a,arc->rc", atax, msh_rc)
    dt = ((dpbase * ptax[None, None, :]).reshape(n, K * C)
          + (dol * taxed[None, :, None, :]).sum(axis=1).reshape(n, K * C))
    system = np.eye(K * C) - B
    rhs = dMA @ IO.T + dhh_by.reshape(n, -1) + dgq.reshape(n, -1)
    if fixed_inv:
        inv_q = qty["inv_channel_q"]
    else:
        tA = float(t @ A)
        system -= np.outer(v.ravel(), t)
        rhs += (dv.reshape(n, -1) * (available0 + tA)
                + v.ravel()[None] * (davail0 + dt @ A)[:, None])
    dA = np.linalg.solve(system, rhs.T).T                 # (n, K*C)
    davail = davail0 + dt @ A + dA @ t
    A = A.reshape(K, C)
    dA = dA.reshape(n, K, C)
    dsavings = (dpq * inv_q[None]).sum(axis=(1, 2)) if fixed_inv else davail

    # ------------------------------------------------------------------
    # Output, factor demand, the current account.
    # ------------------------------------------------------------------
    D = phi * (dsh * arm * A)[None]
    dD = dphi * (dsh * arm * A)[None, None] + phi[None] * ((ddsh * A + dsh * dA) * arm)[:, None]
    carried = float((coef * D).sum())
    dcarried = (coef[None] * dD).sum(axis=(1, 2, 3))
    num_q = D.sum(axis=1) + ss * carried
    dnum_q = dD.sum(axis=2) + dss * carried + ss[None] * dcarried[:, None, None]
    Q_rc = num_q / dfr
    dQ_rc = dnum_q / dfr[None] - Q_rc[None] * ddf / dfr[None]
    dE_rc = def_ * Q_rc[None] + exp_frac[None] * dQ_rc
    E_rc = exp_frac * Q_rc
    M_c = msh * arm * A
    dM_c = (dmsh * A[None] + msh[None] * dA) * arm
    dQ = np.zeros((n, econ.n_com))
    dQ[:, reg.origin_index] = dQ_rc
    Q = np.zeros(econ.n_com)
    Q[reg.origin_index] = Q_rc
    X = econ.msh @ Q
    dX = dQ @ econ.msh.T
    sig_va = (np.ones(na) if econ.spec.value_added == "cobb_douglas"
              else np.broadcast_to(np.asarray(econ.sigma_va, float), (na,)))
    sig_va = np.where(econ.leontief, 0.0, sig_va)
    dfi = sig_va[None, None, :] * fi[None] * (
        _safe_div(dpva, pva[None])[:, None, :] - _safe_div(dw, w[None])[:, :, None])
    dfi = np.where(fi[None] > 0, dfi, 0.0)
    dfac_dem = (dfi * (vunit * X)[None, None, :] + fi[None] * (vunit[None] * dX)[:, None, :]).sum(axis=2)
    if reg.cet:
        ep = np.tile(pe, (R, 1))
        dep = np.broadcast_to(dpe[:, None, :], (n, R, C))
    else:
        ep = p0
        dep = dp0
    dtrade = ((dpm * M_c.sum(axis=0)[None]).sum(axis=1) + (pm[None] * dM_c.sum(axis=1)).sum(axis=1)
              - (dep * E_rc[None] + ep[None] * dE_rc).sum(axis=(1, 2)))
    dcurrent = dtrade + dfac_inc @ fac_row

    # ------------------------------------------------------------------
    # The residual rows, in the order solve_regions writes them.
    # ------------------------------------------------------------------
    dgaps = dfac_dem / np.maximum(np.abs(supply), 1.0)[None]
    for local, (idx, dS) in migrated.items():
        dgaps[:, idx] = ((dfac_dem[:, idx] - dS.sum(axis=1))
                         / np.maximum(np.abs(supply[idx]), 1.0)[None])
    scale = max(abs(ctx["foreign_target"]), 1.0)
    rows = [dgaps[:, i] for i in ctx["others"]]
    rows.append(dcurrent / scale)
    if fixed_inv:
        rows.append((davail - dsavings) / scale)
    return np.array(rows)
