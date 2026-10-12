"""Several regions inside one equilibrium.

The nest added here is one CES, under the Armington the one-region model already
has. Buyers in every region pay for a national composite. That composite is a CES
of the regions of origin, and the Armington then mixes it with the foreign import.
With a single origin the CES price is that origin's price, so a one-region SAM
written with a region label solves as the national model.

What is deliberately not here: factor migration, a regional government closure,
margins, CET exports, and the JuMP replica. Factors clear where they are. There
is one national numeraire, and its market stays out of the system so the Walras
residual is still the check that a sale in one region was a purchase in another.

Bilateral cells of the SAM are not a prediction of the model. The origin mix is
national, common to every buyer. The base year reproduces each region's output
and the national absorption. A destination's purchases from an origin are that
common mix times the destination's own absorption (its households and its
activities). Government and investment buy the composite and are not given a
region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .functions import armington, ces_intensity, ces_price
from .solve import PeriodResult, factor_intensity, value_added_price

__all__ = ["Regional", "build_regional", "solve_regions"]

_PRICE_ITERS = 200


@dataclass
class Regional:
    """The maps the interregional nest adds to a calibrated economy."""

    regions: tuple
    local_commodities: list
    #: Commodity index of origin ``(region, local)``.
    origin_index: np.ndarray
    #: Base-year share of each origin in domestic sales of the local commodity.
    origin_share: np.ndarray          # (n_reg, n_local)
    sigma: np.ndarray                 # (n_local,)
    dom_share: np.ndarray             # national Armington domestic share
    arm_share: np.ndarray
    ptax: np.ndarray
    sigma_arm: np.ndarray
    hh_budget: np.ndarray             # (n_hh, n_local)
    gov_dem: np.ndarray               # purchaser values
    inv_dem: np.ndarray
    io: np.ndarray                    # (n_local, n_act), basic units
    exp_share: np.ndarray             # (n_reg, n_local)
    activity_region: np.ndarray       # (n_act,) int
    household_region: np.ndarray      # (n_hh,) int

    @property
    def n_reg(self) -> int:
        return len(self.regions)

    @property
    def n_local(self) -> int:
        return len(self.local_commodities)


def _split(name: str, regions: tuple) -> tuple[str, str]:
    reg, sep, local = name.partition(":")
    if not sep or reg not in regions:
        raise ValueError(
            f"account {name!r} is in a regional model but is not 'region:name' "
            f"with region one of {list(regions)}")
    return reg, local


def build_regional(econ, sam) -> Regional:
    """Read the origin nest off a calibrated regional economy. Refuse a SAM it cannot price."""
    spec = econ.spec
    regions = tuple(spec.regions)
    acts, coms = econ.acts, econ.coms
    local: list[str] = []
    for name in coms:
        _, loc = _split(name, regions)
        if loc not in local:
            local.append(loc)
    missing = [f"{reg}:{loc}" for reg in regions for loc in local
               if f"{reg}:{loc}" not in coms]
    if missing:
        raise ValueError(
            "every region carries the same commodities. Missing origins, which "
            f"would otherwise be a silent zero share: {missing}")
    r_of = {r: i for i, r in enumerate(regions)}
    c_of = {c: i for i, c in enumerate(local)}
    origin_index = np.empty((len(regions), len(local)), dtype=int)
    for i, name in enumerate(coms):
        reg, loc = _split(name, regions)
        origin_index[r_of[reg], c_of[loc]] = i

    # An activity produces in its own region. A make cell across regions would
    # be a second theory of trade, on top of the origin CES.
    make = sam.block(acts, coms)
    activity_region = np.empty(len(acts), dtype=int)
    for j, name in enumerate(acts):
        reg, _ = _split(name, regions)
        activity_region[j] = r_of[reg]
        for k, com in enumerate(coms):
            if abs(make[j, k]) > 1e-12:
                creg, _ = _split(com, regions)
                if creg != reg:
                    raise ValueError(
                        f"activity {name!r} produces {com!r}, in another region. "
                        "Trade goes through the origin nest, not through the make matrix.")
    household_region = np.array([r_of[_split(h, regions)[0]] for h in econ.hhs], dtype=int)
    for name in econ.factors:
        _split(name, regions)

    if np.any(np.abs(econ.mq) > 1e-12):
        raise ValueError(
            "a regional SAM with trade and transport margins is not in this slice. "
            "The commodity-commodity block would mix margins with origin trade.")
    if econ.tm0 is not None or econ.tauz0 is not None:
        raise ValueError(
            "the regional nest does not apply a tariff or an indirect tax on "
            "output yet. Calibrate those on a one-region model.")
    if econ.own_account.any():
        raise ValueError(
            "own-account activities are not in the regional slice. An activity "
            "that sells straight to households has no origin price.")
    if spec.export_rule != "fixed_shares":
        raise ValueError(
            "regional exports in this slice are the base-year share of output. "
            f"export_rule={spec.export_rule!r} would price origins twice, once on "
            "the CET frontier and once in the origin CES.")
    if spec.energy:
        raise ValueError(
            "the regional slice keeps intermediate inputs Leontief; an energy nest "
            "(ModelSpec.energy) is a one-region feature for now.")
    if spec.export_demand is not None or spec.import_supply is not None:
        raise ValueError(
            "the regional slice is a price taker abroad. export_demand and "
            "import_supply would need one world price per national commodity, "
            "and the origin nest does not carry it.")
    if spec.household_demand != "cobb_douglas":
        raise ValueError(
            "regional households buy the national composite with Cobb-Douglas "
            f"shares. household_demand={spec.household_demand!r} is not in this slice.")

    n_reg, n_loc = len(regions), len(local)
    dom = np.zeros((n_reg, n_loc))
    imp = np.zeros((n_reg, n_loc))
    exp_share = np.zeros((n_reg, n_loc))
    for r in range(n_reg):
        for c in range(n_loc):
            i = origin_index[r, c]
            dom[r, c] = econ.dom0[i]
            imp[r, c] = econ.imp0[i]
            exp_share[r, c] = econ.exp_share[i]
    dom_total = dom.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        origin_share = np.where(dom_total > 0, dom / np.maximum(dom_total, 1e-12), 0.0)
    absorbed = dom_total + imp.sum(axis=0)
    dom_share = np.where(absorbed > 0, dom_total / np.maximum(absorbed, 1e-12), 1.0)
    # One tax rate on the composite. Different origin rates would be a wedge on
    # the origin price, which this slice does not have; pooling them would move
    # the mix for a reason that is not the elasticity.
    ptax_o = np.zeros((n_reg, n_loc))
    for r in range(n_reg):
        for c in range(n_loc):
            ptax_o[r, c] = econ.ptax[origin_index[r, c]]
    for c, name in enumerate(local):
        rates = ptax_o[:, c]
        if np.max(np.abs(rates - rates[0])) > 1e-8:
            raise ValueError(
                f"commodity {name!r} has different product-tax rates across regions "
                f"({rates.tolist()}). This slice puts one rate on the national composite.")
    ptax = ptax_o[0].copy()

    sigma = _sigma(spec, local)
    sigma_arm = np.array([_arm_sigma(spec, econ, origin_index[:, c]) for c in range(n_loc)])

    spend = econ.hh_budget * np.asarray(econ.hh_spend0, float)[:, None]
    nat_spend = np.zeros((econ.n_hh, n_loc))
    nat_io = np.zeros((n_loc, econ.n_act))
    nat_gov = np.zeros(n_loc)
    nat_inv = np.zeros(n_loc)
    for c in range(n_loc):
        cols = origin_index[:, c]
        nat_spend[:, c] = spend[:, cols].sum(axis=1)
        nat_io[c] = econ.io[cols].sum(axis=0)
        nat_gov[c] = econ.gov_dem[cols].sum()
        nat_inv[c] = econ.inv_dem[cols].sum()
    nat_budget = nat_spend / np.maximum(nat_spend.sum(axis=1, keepdims=True), 1e-12)

    return Regional(
        regions=regions, local_commodities=local, origin_index=origin_index,
        origin_share=origin_share, sigma=sigma, dom_share=dom_share,
        arm_share=np.ones(n_loc), ptax=ptax, sigma_arm=sigma_arm,
        hh_budget=nat_budget, gov_dem=nat_gov, inv_dem=nat_inv, io=nat_io,
        exp_share=exp_share, activity_region=activity_region,
        household_region=household_region)


def _sigma(spec, local: list[str]) -> np.ndarray:
    value = spec.sigma_interregional
    if isinstance(value, dict):
        missing = [n for n in local if n not in value]
        if missing:
            raise KeyError(f"sigma_interregional has no entry for {missing}")
        out = np.array([float(value[n]) for n in local])
    else:
        out = np.full(len(local), float(value))
    if np.any(out < 0):
        raise ValueError(f"sigma_interregional must be >= 0, got {out}")
    return out


def _arm_sigma(spec, econ, origin_cols: np.ndarray) -> float:
    """The Armington elasticity of a local commodity.

    A scalar on the spec is that elasticity. A mapping is keyed by the local
    name or by any origin account, and the origins of one commodity must agree.
    """
    value = spec.sigma_armington
    if not isinstance(value, dict):
        return float(value)
    rates = []
    for i in origin_cols:
        name = econ.coms[i]
        local = name.split(":", 1)[1]
        if local in value:
            rates.append(float(value[local]))
        elif name in value:
            rates.append(float(value[name]))
        elif "__default__" in value:
            rates.append(float(value["__default__"]))
        else:
            raise KeyError(f"sigma_armington has no entry for {name!r} or {local!r}")
    if max(rates) - min(rates) > 1e-12:
        raise ValueError(
            f"origins of one commodity have different Armington elasticities: {rates}")
    return rates[0]


def solve_regions(econ, closure, scenario, *, warm_start=None, raise_on_fail: bool = True):
    """Solve the interregional nest. The legacy solver does not see these accounts."""
    reg: Regional = econ.regional
    _refuse_unsupported(econ, closure, scenario, reg)
    nf = econ.n_fac
    num = econ.numeraire_index
    others = [i for i in range(nf) if i != num]
    i_er = len(others)
    n_unknown = i_er + 1
    nu = closure.numeraire
    supply = (econ.fac_supply if scenario.factor_supply is None
              else np.asarray(scenario.factor_supply, float))
    if supply.shape != (nf,):
        raise ValueError(
            f"factor_supply must have length {nf}, got shape {supply.shape}")
    tfp = np.ones(econ.n_act) if scenario.productivity is None else np.asarray(
        scenario.productivity, float)
    if tfp.shape != (econ.n_act,):
        raise ValueError(
            f"productivity must have length {econ.n_act}, got shape {tfp.shape}")
    transfers = (np.zeros(econ.n_hh) if scenario.transfers is None
                 else np.asarray(scenario.transfers, float))
    world = np.ones(reg.n_local) if scenario.world_price is None else np.asarray(
        scenario.world_price, float)
    if world.shape != (reg.n_local,):
        raise ValueError(
            "world_price in a regional model is one foreign price per local "
            f"commodity ({reg.local_commodities}), got shape {world.shape}")
    facshare = (econ.hh_facshare if scenario.factor_ownership is None
                else np.asarray(scenario.factor_ownership, float))
    fac_gov = econ.zeros_if_none("fac_gov_share", nf)
    fac_row = econ.zeros_if_none("fac_row_share", nf)
    dtax = econ.zeros_if_none("hh_dtax", econ.n_hh)
    gov_tr = econ.zeros_if_none("hh_gov_transfer", econ.n_hh) * nu
    remit = econ.zeros_if_none("hh_remit_net", econ.n_hh) * nu
    foreign_target = econ.foreign_sav * nu
    state: dict = {}

    def residuals(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        w = np.full(nf, nu)
        for k, i in enumerate(others):
            w[i] = nu * float(np.exp(z[k]))
        er = float(np.exp(z[i_er]))
        prices = _prices(econ, reg, w, er, world, tfp, nu)
        qty = _quantities(econ, reg, w, supply, facshare, fac_gov, fac_row, dtax,
                          transfers, gov_tr, remit, foreign_target, tfp, prices, nu)
        gaps = (qty["fac_dem"] - supply) / np.maximum(np.abs(supply), 1.0)
        out = [float(gaps[i]) for i in others]
        scale = max(abs(foreign_target), 1.0)
        out.append((qty["current"] - foreign_target) / scale)
        state.update(prices)
        state.update(qty)
        state.update(w=w, er=er, gaps=gaps, walras=float(gaps[num]))
        return np.asarray(out, dtype=float)

    z_hat, residual, label, message = _solve(residuals, n_unknown, closure.tolerance,
                                             warm_start, raise_on_fail)
    residuals(z_hat)
    return _pack(econ, reg, closure, scenario, state, z_hat, residual, label, message,
                 closure.tolerance)


def _refuse_unsupported(econ, closure, scenario, reg: Regional):
    if closure.labour != "full_employment":
        raise ValueError(
            "regional factors clear where they are supplied. "
            f"closure.labour={closure.labour!r} would move a wage without a "
            "migration equation, and the two are different closures.")
    if closure.investment != "savings_driven" or closure.government != "real_fixed":
        raise ValueError(
            "the regional slice solves savings-driven investment and government "
            f"quantities fixed. Got investment={closure.investment!r}, "
            f"government={closure.government!r}.")
    if closure.foreign != "fixed_saving":
        raise ValueError(
            "the exchange rate clears one national current account. "
            f"closure.foreign={closure.foreign!r} is not in this slice.")
    blocked = {
        "consumer_wedge": scenario.consumer_wedge,
        "government_wedge": scenario.government_wedge,
        "product_tax": scenario.product_tax,
        "product_tax_firm": scenario.product_tax_firm,
        "export_price": scenario.export_price,
        "investment_mix": scenario.investment_mix,
        "foreign_factor_share": scenario.foreign_factor_share,
    }
    named = [k for k, v in blocked.items() if v is not None]
    if named or scenario.activity_scale or scenario.fdi:
        extra = list(named)
        if scenario.activity_scale:
            extra.append("activity_scale")
        if scenario.fdi:
            extra.append("fdi")
        raise ValueError(
            "the regional slice does not read these scenario fields, and will not "
            f"ignore them: {extra}")
    if reg.n_reg < 1:
        raise ValueError("regional model has no regions")


def _prices(econ, reg: Regional, w, er, world, tfp, nu):
    """Origin prices, the national domestic CES, then Armington against imports."""
    io = reg.io / tfp[None, :]
    vunit = econ.vunit / tfp
    # Specific activity tax is denominated in the numeraire, same as the one-region solver.
    atax = econ.atax * nu
    pq = np.ones(reg.n_local)
    p_origin = np.ones(econ.n_com)
    pa = np.ones(econ.n_act)
    pd_nat = np.ones(reg.n_local)
    phi = reg.origin_share.copy()
    dsh = reg.dom_share.copy()
    msh = 1.0 - dsh
    for _ in range(_PRICE_ITERS):
        pva = value_added_price(econ, w)
        pa = io.T @ pq + pva * vunit + atax
        p_origin = econ.msh.T @ pa
        p_rc = p_origin[reg.origin_index]
        pd_nat = np.zeros(reg.n_local)
        phi = np.zeros_like(reg.origin_share)
        for c in range(reg.n_local):
            sh = reg.origin_share[:, c:c + 1]
            pd_nat[c] = float(ces_price(sh, p_rc[:, c], reg.sigma[c])[0])
            phi[:, c] = ces_intensity(
                sh, p_rc[:, c], np.array([pd_nat[c]]), reg.sigma[c])[:, 0]
        pm = er * world
        pbase, dsh, msh = armington(reg.dom_share, pd_nat, pm, reg.sigma_arm)
        new = pbase * (1.0 + reg.ptax)
        if np.max(np.abs(new - pq)) < 1e-14:
            pq = new
            break
        pq = new
    return dict(pa=pa, p_origin=p_origin, pd_nat=pd_nat, phi=phi, pq=pq,
                pbase=pq / np.maximum(1.0 + reg.ptax, 1e-12), dsh=dsh, msh=msh,
                pm=er * world)


def _quantities(econ, reg, w, supply, facshare, fac_gov, fac_row, dtax, transfers,
                gov_tr, remit, foreign_target, tfp, prices, nu):
    """National absorption, then each origin's sales. Tax revenue is affine in absorption."""
    C = reg.n_local
    pq = prices["pq"]
    phi = prices["phi"]
    dsh = prices["dsh"]
    fac_inc = w * supply
    kept = fac_inc * (1.0 - fac_gov - fac_row)
    hh_income = facshare @ kept + transfers + gov_tr + remit
    disposable = hh_income * (1.0 - dtax)
    hh_sav = disposable * econ.hh_savrate
    spend = disposable - hh_sav
    hh_q = reg.hh_budget * spend[:, None] / np.maximum(pq, 1e-12)
    hh_final = hh_q.sum(axis=0)
    gov_q = reg.gov_dem / np.maximum(1.0 + reg.ptax, 1e-12)
    inv_share = reg.inv_dem / max(float(reg.inv_dem.sum()), 1e-12) if reg.inv_dem.sum() > 0 \
        else np.zeros(C)
    gov_spend = float(pq @ gov_q)
    gov_fixed = float((fac_inc * fac_gov).sum()) + float((hh_income * dtax).sum()) - gov_spend \
        - float(gov_tr.sum())
    S0 = float(hh_sav.sum()) + gov_fixed + foreign_target
    v = inv_share / np.maximum(pq, 1e-12)          # inv quantity per unit of nominal saving
    F0 = hh_final + gov_q + v * S0

    scale = phi / np.maximum(1.0 - reg.exp_share, 1e-12)   # Q per unit of national domestic sales
    link = scale * dsh[None, :]                             # (R, C): Q per unit of A
    # X = msh @ Q_full, Q_full[origin] = link[r,c] * A[c]
    M = np.zeros((econ.n_act, C))
    for r in range(reg.n_reg):
        for c in range(C):
            i = reg.origin_index[r, c]
            M[:, c] += econ.msh[:, i] * link[r, c]
    io = reg.io / tfp[None, :]
    B = io @ M                                          # A -> intermediate use
    # Product tax on absorption, activity tax on output. Output is M @ A.
    tax_per = prices["pbase"] * reg.ptax + M.T @ (econ.atax * nu)
    eye = np.eye(C)
    try:
        L = np.linalg.solve(eye - B, eye)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "the regional quantity system is singular. A commodity is used in "
            "its own production with a coefficient of one or more.") from exc
    # A = L @ (F0 + outer(v, tax) @ A)
    leak = L @ np.outer(v, tax_per)
    try:
        A = np.linalg.solve(eye - leak, L @ F0)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "product-tax revenue feeds investment which feeds the tax base, and "
            "that loop does not have a finite sum.") from exc
    savings = S0 + float(tax_per @ A)
    inv_q = v * savings
    ND = dsh * A
    Q_rc = link * A[None, :]
    D_rc = phi * ND[None, :]
    E_rc = reg.exp_share * Q_rc
    M_c = prices["msh"] * A
    Q = np.zeros(econ.n_com)
    exports = np.zeros(econ.n_com)
    imports = np.zeros(econ.n_com)
    for r in range(reg.n_reg):
        for c in range(C):
            i = int(reg.origin_index[r, c])
            Q[i] = Q_rc[r, c]
            exports[i] = E_rc[r, c]
            # Imports are national. Spread them by the base-year import weights
            # so the regional import vector still sums to the national one.
    imp0 = np.array([econ.imp0[reg.origin_index[r]] for r in range(reg.n_reg)])
    imp_w = imp0 / np.maximum(imp0.sum(axis=0), 1e-12)
    imp_w = np.where(imp0.sum(axis=0) > 0, imp_w, 0.0)
    for r in range(reg.n_reg):
        for c in range(C):
            imports[reg.origin_index[r, c]] = imp_w[r, c] * M_c[c]
    X = econ.msh @ Q
    # Own-account was refused, so the make matrix recovers every activity.
    fi = factor_intensity(econ, w, vunit=econ.vunit / tfp, aunit=econ.aunit / tfp[None, :])
    fac_by = fi * ((econ.vunit / tfp) * X)[None, :]
    fac_dem = fac_by.sum(axis=1)

    hh_by = np.zeros((reg.n_reg, C))
    for h in range(econ.n_hh):
        hh_by[reg.household_region[h]] += hh_q[h]
    inter = np.zeros((reg.n_reg, C))
    use = (io * X[None, :])
    for j in range(econ.n_act):
        inter[reg.activity_region[j]] += use[:, j]
    dest = hh_by + inter
    bilateral = phi[:, None, :] * dest[None, :, :]

    trade = float(prices["pm"] @ M_c - prices["p_origin"] @ exports)
    current = trade + float((fac_inc * fac_row).sum()) - float(remit.sum())
    # Household quantities on the regional grid: the origin mix times the
    # household's composite quantity. They sum back to the composite.
    hh_origin = np.zeros((econ.n_hh, econ.n_com))
    gov_origin = np.zeros(econ.n_com)
    inv_origin = np.zeros(econ.n_com)
    for r in range(reg.n_reg):
        for c in range(C):
            i = int(reg.origin_index[r, c])
            hh_origin[:, i] = phi[r, c] * hh_q[:, c]
            gov_origin[i] = phi[r, c] * gov_q[c]
            inv_origin[i] = phi[r, c] * inv_q[c]
    return dict(
        A=A, Q=Q, X=X, D=D_rc, exports=exports, imports=imports, hh_q=hh_origin,
        hh_income=hh_income, spend=spend, gov_q=gov_origin, inv_q=inv_origin,
        fac_dem=fac_dem, fac_by=fac_by, savings=savings, current=current,
        bilateral=bilateral, hh_nat=hh_q, gov_nat=gov_q, inv_nat=inv_q, ND=ND)


def _solve(residuals, n_unknown, tolerance, warm_start, raise_on_fail):
    attempts = []
    if warm_start is not None:
        z0 = getattr(warm_start, "z", warm_start)
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
        except (ValueError, RuntimeError):
            continue
        res = float(np.max(np.abs(residuals(sol.x))))
        if best is None or res < best[0]:
            best = (res, np.asarray(sol.x, dtype=float), label, str(sol.message))
        if res < tolerance:
            break
    if best is None:
        raise RuntimeError("solve_period: every start point failed before a residual was formed")
    res, z, label, message = best
    if res >= tolerance and raise_on_fail:
        raise RuntimeError(
            f"solve_period: residual {res:.3e} above tolerance {tolerance:.1e} "
            f"({label}). Solver message: {message}")
    return z, res, label, message


def _pack(econ, reg, closure, scenario, state, z, residual, label, message, tolerance):
    pq_full = np.zeros(econ.n_com)
    pbase_full = np.zeros(econ.n_com)
    for r in range(reg.n_reg):
        for c in range(reg.n_local):
            i = int(reg.origin_index[r, c])
            pq_full[i] = state["pq"][c]
            pbase_full[i] = state["pbase"][c]
    yard = 1.0 + reg.ptax
    # Basket cost relative to the base year, weights fixed at the calibrated shares.
    base_cost = reg.hh_budget @ yard
    now_cost = reg.hh_budget @ state["pq"]
    hh_index = np.where(base_cost > 0, now_cost / np.maximum(base_cost, 1e-12), 1.0)
    ok = residual < tolerance
    return PeriodResult(
        econ=econ, closure=closure, scenario=scenario, z=z,
        factor_prices=state["w"], exchange_rate=state["er"],
        pa=state["pa"], pd=state["p_origin"], pq=pq_full, pqf=pq_full, pbase=pbase_full,
        Q=state["Q"], X=state["X"], absorption=_absorb(reg, state),
        imports=state["imports"], exports=state["exports"],
        hh_quantity=state["hh_q"], hh_income=state["hh_income"],
        hh_commodity_spend=state["spend"],
        hh_direct_outlay=np.zeros(econ.n_hh), hh_wedge_paid=np.zeros(econ.n_hh),
        gov_quantity=state["gov_q"], investment_quantity=state["inv_q"],
        factor_demand=state["fac_dem"], factor_demand_by_activity=state["fac_by"],
        savings=state["savings"], wedge_revenue=0.0, government_cost=0.0,
        real=dict(
            hh_real=float(yard @ state["hh_nat"].sum(axis=0)),
            gov_real=float(yard @ state["gov_nat"]),
            inv_real=float(yard @ state["inv_nat"]),
            hh_real_income=np.where(hh_index > 0, state["spend"] / hh_index, 0.0),
            hh_price_index=hh_index,
            national_absorption=state["A"],
            national_price=state["pq"],
        ),
        gaps={f: float(g) for f, g in zip(econ.factors, state["gaps"])},
        walras=state["walras"], residual=residual, converged=ok, attempt=label,
        message=message, bilateral=state["bilateral"])


def _absorb(reg, state):
    """Regional domestic sales plus the imports attributed to that origin."""
    out = np.zeros(state["Q"].shape)
    for r in range(reg.n_reg):
        for c in range(reg.n_local):
            i = int(reg.origin_index[r, c])
            out[i] = state["D"][r, c] + state["imports"][i]
    return out
