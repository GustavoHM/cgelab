"""Several regions inside one equilibrium.

The nest added here is one CES, under the Armington the one-region model already
has. Buyers in every region pay for a national composite. That composite is a CES
of the regions of origin, and the Armington then mixes it with the foreign import.
With a single origin the CES price is that origin's price, so a one-region SAM
written with a region label solves as the national model.

Delivered trade and transport margins enter both purchaser prices and the
quantity system. Under ``export_rule="cet"``, each origin splits output between
home sales and exports in response to the foreign price. A regional government
closure and the JuMP replica are not here. There is one national numeraire, and
its market stays out of the system so the Walras residual remains an independent
check that a sale in one region was a purchase in another.

Bilateral cells of the SAM are not a prediction of the model. The origin mix is
national, common to every buyer. The base year reproduces each region's output
and the national absorption. A destination's purchases from an origin are that
common mix times the destination's own absorption (its households and its
activities). Government and investment buy the composite, with optional
additional accounts mapped by region.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from .functions import armington, ces_intensity, ces_price, cet, cet_domestic_price
from .mobility import FactorMobility
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
    sigma_cet: np.ndarray             # (n_reg, n_local)
    mq: np.ndarray                    # (n_local, n_local), margin per absorption
    cet: bool
    activity_region: np.ndarray       # (n_act,) int
    household_region: np.ndarray      # (n_hh,) int
    government_accounts: dict = field(default_factory=dict)
    investment_accounts: dict = field(default_factory=dict)
    gov_dem_regional: np.ndarray | None = None  # (n_reg, n_local), zero absent
    inv_dem_regional: np.ndarray | None = None
    fac_gov_regional: np.ndarray | None = None  # (n_reg, n_fac), zero absent
    migration: dict = field(default_factory=dict)

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


def _institution_demand(econ, sam, mapping, regions, origin_index, label):
    """Regional institution demand, with no arbitrary split of national totals."""
    accounts = dict(mapping or {})
    unknown_regions = sorted(set(accounts) - set(regions))
    if unknown_regions:
        raise ValueError(f"{label} names regions not in the model: {unknown_regions}")
    if len(set(accounts.values())) != len(accounts):
        raise ValueError(f"{label} uses one account for more than one region")
    missing_accounts = sorted(set(accounts.values()) - set(sam.accounts))
    if missing_accounts:
        raise KeyError(f"{label} names accounts not in the SAM: {missing_accounts}")
    demand = np.zeros(origin_index.shape)
    for i, region in enumerate(regions):
        if region in accounts:
            account = accounts[region]
            demand[i] = sam.block(list(econ.coms), [account]).ravel()[origin_index[i]]
    return accounts, demand


def build_regional(econ, sam, *, migration_pools=None) -> Regional:
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

    if econ.tm0 is not None or econ.tauz0 is not None:
        raise ValueError(
            "the regional nest does not apply a tariff or an indirect tax on "
            "output yet. Calibrate those on a one-region model.")
    if econ.own_account.any():
        raise ValueError(
            "own-account activities are not in the regional slice. An activity "
            "that sells straight to households has no origin price.")
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
    sigma_cet = np.array([_arm_sigma(spec, econ, origin_index[:, c], "sigma_cet")
                          for c in range(n_loc)])

    spend = econ.hh_budget * np.asarray(econ.hh_spend0, float)[:, None]
    nat_spend = np.zeros((econ.n_hh, n_loc))
    nat_io = np.zeros((n_loc, econ.n_act))
    nat_gov = np.zeros(n_loc)
    nat_inv = np.zeros(n_loc)
    nat_a = np.zeros(n_loc)
    nat_arm = np.zeros(n_loc)
    nat_mq = np.zeros((n_loc, n_loc))
    for c in range(n_loc):
        cols = origin_index[:, c]
        nat_spend[:, c] = spend[:, cols].sum(axis=1)
        nat_io[c] = econ.io[cols].sum(axis=0)
        nat_gov[c] = econ.gov_dem[cols].sum()
        nat_inv[c] = econ.inv_dem[cols].sum()
        nat_a[c] = econ.A0[cols].sum()
        nat_arm[c] = (econ.arm_share[cols] * econ.A0[cols]).sum()
        for k in range(n_loc):
            rows = origin_index[:, k]
            nat_mq[k, c] = (econ.mq[rows][:, cols] * econ.A0[cols]).sum()
    nat_arm /= np.maximum(nat_a, 1e-12)
    nat_mq /= np.maximum(nat_a, 1e-12)[None, :]
    nat_budget = nat_spend / np.maximum(nat_spend.sum(axis=1, keepdims=True), 1e-12)
    government_accounts, gov_regional = _institution_demand(
        econ, sam, spec.regional_government, regions, origin_index, "regional_government")
    investment_accounts, inv_regional = _institution_demand(
        econ, sam, spec.regional_investment, regions, origin_index, "regional_investment")
    fac_gov_regional = np.zeros((n_reg, econ.n_fac))
    if government_accounts:
        hh_fac = sam.block(econ.hhs, econ.factors)
        other_gov = sum((sam.block([account], econ.factors).ravel()
                         for account in government_accounts.values()),
                        start=np.zeros(econ.n_fac))
        fac_total = (hh_fac.sum(axis=0) + other_gov
                     + sam.block([econ.layout.government], econ.factors).ravel()
                     + sam.block([econ.layout.rest_of_world], econ.factors).ravel())
        for region, account in government_accounts.items():
            i = regions.index(region)
            payments = sam.block([account], econ.factors).ravel()
            fac_gov_regional[i] = np.where(fac_total > 0,
                                           payments / np.maximum(fac_total, 1e-12), 0.0)

    migration = _build_migration(spec, econ, regions, migration_pools)

    return Regional(
        regions=regions, local_commodities=local, origin_index=origin_index,
        origin_share=origin_share, sigma=sigma, dom_share=dom_share,
        arm_share=nat_arm, ptax=ptax, sigma_arm=sigma_arm,
        hh_budget=nat_budget, gov_dem=nat_gov, inv_dem=nat_inv, io=nat_io,
        exp_share=exp_share, sigma_cet=np.repeat(sigma_cet[None, :], n_reg, axis=0),
        mq=nat_mq, cet=spec.export_rule == "cet", activity_region=activity_region,
        household_region=household_region, government_accounts=government_accounts,
        investment_accounts=investment_accounts,
        gov_dem_regional=gov_regional, inv_dem_regional=inv_regional,
        fac_gov_regional=fac_gov_regional, migration=migration)


def _build_migration(spec, econ, regions, migration_pools=None):
    """CET/logit migration pools by local factor, with origin income preserved."""
    pools_in = dict(migration_pools or {})
    out = {}
    for local, omega in (spec.factor_migration or {}).items():
        omega = float(omega)
        if not np.isfinite(omega) or omega < 0:
            raise ValueError(f"factor_migration[{local!r}] must be finite and >= 0")
        names = [f"{region}:{local}" for region in regions]
        missing = [name for name in names if name not in econ.factors]
        if missing:
            raise KeyError(f"factor_migration[{local!r}] needs these regional factors: {missing}")
        indices = np.array([econ.factors.index(name) for name in names])
        endowment = np.asarray(econ.fac_supply[indices], dtype=float)
        if local in pools_in:
            frame = pools_in[local]
            if set(frame.index) != set(regions) or set(frame.columns) != set(regions):
                raise ValueError(f"migration pool {local!r} must index and column every region")
            delta = frame.loc[list(regions), list(regions)].to_numpy(dtype=float)
        else:
            delta = np.eye(len(regions))
        if not np.isfinite(delta).all() or (delta < 0).any():
            raise ValueError(f"migration pool {local!r} must be finite and nonnegative")
        if not np.allclose(delta.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"migration pool {local!r} rows must sum to one")
        if not np.allclose(endowment @ delta, endowment, rtol=1e-10, atol=1e-10):
            raise ValueError(f"migration pool {local!r} does not reproduce regional endowments")
        form_option = spec.factor_mobility_form
        form = form_option.get(local, form_option.get("__default__", "cet")) \
            if isinstance(form_option, dict) else form_option
        out[local] = {
            "model": FactorMobility(factor=local, index=int(indices[0]), form=form,
                                    omega=omega, users=np.ones(len(regions), dtype=bool),
                                    pools=list(regions), delta=delta,
                                    endowment0=endowment, allow_new=True),
            "indices": indices,
        }
    return out


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


def _arm_sigma(spec, econ, origin_cols: np.ndarray, field="sigma_armington") -> float:
    """One elasticity per local commodity, with agreeing regional entries."""
    value = getattr(spec, field)
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
            raise KeyError(f"{field} has no entry for {name!r} or {local!r}")
    if max(rates) - min(rates) > 1e-12:
        raise ValueError(f"origins of one commodity have different {field} values: {rates}")
    return rates[0]


def solve_regions(econ, closure, scenario, *, warm_start=None, raise_on_fail: bool = True):
    """Solve the interregional nest. The legacy solver does not see these accounts."""
    reg: Regional = econ.regional
    _refuse_unsupported(econ, closure, scenario, reg)
    nf = econ.n_fac
    num = econ.numeraire_index
    others = [i for i in range(nf) if i != num]
    i_er = len(others)
    i_inv = i_er + 1
    fixed_investment = closure.investment == "fixed_real"
    n_unknown = i_er + 1 + int(fixed_investment)
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
    export_world = (np.ones(reg.n_local) if scenario.export_price is None
                    else np.asarray(scenario.export_price, float))
    if export_world.shape != (reg.n_local,):
        raise ValueError(
            "export_price in a regional model is one foreign price per local "
            f"commodity ({reg.local_commodities}), got shape {export_world.shape}")
    if scenario.export_price is not None and not reg.cet:
        raise ValueError("Scenario.export_price requires export_rule='cet' in a regional model")
    facshare = (econ.hh_facshare if scenario.factor_ownership is None
                else np.asarray(scenario.factor_ownership, float))
    fac_gov = econ.zeros_if_none("fac_gov_share", nf)
    fac_row = econ.zeros_if_none("fac_row_share", nf)
    dtax = econ.zeros_if_none("hh_dtax", econ.n_hh)
    gov_tr = econ.zeros_if_none("hh_gov_transfer", econ.n_hh) * nu
    remit = econ.zeros_if_none("hh_remit_net", econ.n_hh) * nu
    foreign_target = econ.foreign_sav * nu
    investment_level = (reg.inv_dem.sum() if scenario.investment_level is None
                        else float(scenario.investment_level))
    if not np.isfinite(investment_level) or investment_level < 0:
        raise ValueError("investment_level must be finite and nonnegative")
    state: dict = {}

    def residuals(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        w = np.full(nf, nu)
        for k, i in enumerate(others):
            w[i] = nu * float(np.exp(z[k]))
        er = float(np.exp(z[i_er]))
        levy = float(z[i_inv]) if fixed_investment else 0.0
        prices = _prices(econ, reg, w, er, world, export_world, tfp, nu)
        qty = _quantities(econ, reg, w, supply, facshare, fac_gov, fac_row, dtax,
                          transfers - levy * econ.pop_share, gov_tr, remit,
                          foreign_target, tfp, prices, nu,
                          investment_level=investment_level, levy=levy,
                          fixed_investment=fixed_investment, migration=reg.migration)
        gaps = (qty["fac_dem"] - supply) / np.maximum(np.abs(supply), 1.0)
        for local, item in reg.migration.items():
            idx = item["indices"]
            migrated = qty["migration"][local]["supply"].sum(axis=0)
            gaps[idx] = ((qty["fac_dem"][idx] - migrated)
                         / np.maximum(np.abs(supply[idx]), 1.0))
        out = [float(gaps[i]) for i in others]
        scale = max(abs(foreign_target), 1.0)
        out.append((qty["current"] - foreign_target) / scale)
        if fixed_investment:
            out.append((qty["available"] - qty["savings"]) / scale)
        state.update(prices)
        state.update(qty)
        state.update(w=w, er=er, gaps=gaps, walras=float(gaps[num]),
                     levy=levy, foreign_sav=foreign_target)
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
    if closure.investment not in ("savings_driven", "fixed_real") or closure.government != "real_fixed":
        raise ValueError(
            "the regional slice solves savings-driven or fixed-real investment and "
            f"government quantities fixed. Got investment={closure.investment!r}, "
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


def _prices(econ, reg: Regional, w, er, world, export_world, tfp, nu):
    """CET origin sales, the domestic CES, Armington, then delivered margins."""
    io = reg.io / tfp[None, :]
    vunit = econ.vunit / tfp
    atax = econ.atax * nu
    pq = np.ones(reg.n_local)
    p_origin = np.ones(econ.n_com)
    pa = np.ones(econ.n_act)
    pd_nat = np.ones(reg.n_local)
    phi = reg.origin_share.copy()
    dsh = reg.dom_share.copy()
    msh = 1.0 - dsh
    pe = er * export_world
    for _ in range(_PRICE_ITERS):
        pva = value_added_price(econ, w)
        pa = io.T @ pq + pva * vunit + atax
        p_origin = econ.msh.T @ pa
        p_rc = p_origin[reg.origin_index]
        if reg.cet:
            p_rc = cet_domestic_price(reg.exp_share, p_rc, pe[None, :], reg.sigma_cet)
        pd_nat = np.zeros(reg.n_local)
        phi = np.zeros_like(reg.origin_share)
        for c in range(reg.n_local):
            sh = reg.origin_share[:, c:c + 1]
            pd_nat[c] = float(ces_price(sh, p_rc[:, c], reg.sigma[c])[0])
            phi[:, c] = ces_intensity(
                sh, p_rc[:, c], np.array([pd_nat[c]]), reg.sigma[c])[:, 0]
        pm = er * world
        arm, dsh, msh = armington(reg.dom_share, pd_nat, pm, reg.sigma_arm)
        pbase = arm * reg.arm_share + pq @ reg.mq
        new = pbase * (1.0 + reg.ptax)
        converged = np.max(np.abs(new - pq)) < 1e-14
        pq = new
        if converged:
            break
    else:
        raise RuntimeError(f"regional price fixed point did not converge in {_PRICE_ITERS} iterations")
    p_origin = econ.msh.T @ pa
    p_rc = p_origin[reg.origin_index]
    if reg.cet:
        p_rc = cet_domestic_price(reg.exp_share, p_rc, pe[None, :], reg.sigma_cet)
        _, dom_frac, exp_frac = cet(reg.exp_share, p_rc, pe[None, :], reg.sigma_cet)
    else:
        dom_frac, exp_frac = 1.0 - reg.exp_share, reg.exp_share
    return dict(pa=pa, p_origin=p_origin, pd_nat=pd_nat, phi=phi, pq=pq,
                pbase=pq / np.maximum(1.0 + reg.ptax, 1e-12), dsh=dsh, msh=msh,
                pm=er * world, pe=pe, dom_frac=dom_frac, exp_frac=exp_frac)


def _quantities(econ, reg, w, supply, facshare, fac_gov, fac_row, dtax, transfers,
                gov_tr, remit, foreign_target, tfp, prices, nu, *,
                investment_level, levy, fixed_investment, migration=None):
    """National absorption, then each origin's sales. Tax revenue is affine in absorption."""
    C = reg.n_local
    pq = prices["pq"]
    phi = prices["phi"]
    dsh = prices["dsh"]
    fac_inc = w * supply
    migration_results = {}
    for local, item in (migration or {}).items():
        idx = item["indices"]
        migrated, price = item["model"].supply(w[idx], supply[idx])
        fac_inc[idx] = price * supply[idx]
        migration_results[local] = dict(pools=item["model"].pools, price=price,
                                        endowment=supply[idx], supply=migrated)
    kept = fac_inc * (1.0 - fac_gov - fac_row - reg.fac_gov_regional.sum(axis=0))
    hh_income = facshare @ kept + transfers + gov_tr + remit
    disposable = hh_income * (1.0 - dtax)
    hh_sav = disposable * econ.hh_savrate
    spend = disposable - hh_sav
    hh_q = reg.hh_budget * spend[:, None] / np.maximum(pq, 1e-12)
    hh_final = hh_q.sum(axis=0)
    gov_regional_q = reg.gov_dem_regional / np.maximum(1.0 + reg.ptax, 1e-12)
    gov_q = reg.gov_dem / np.maximum(1.0 + reg.ptax, 1e-12) + gov_regional_q.sum(axis=0)
    inv_total = reg.inv_dem + reg.inv_dem_regional.sum(axis=0)
    inv_share = inv_total / max(float(inv_total.sum()), 1e-12) if inv_total.sum() > 0 \
        else np.zeros(C)
    gov_spend = float(pq @ gov_q)
    gov_fixed = (float((fac_inc * fac_gov).sum())
                 + float((fac_inc * reg.fac_gov_regional).sum())
                 + float((hh_income * dtax).sum()) - gov_spend - float(gov_tr.sum()))
    available0 = float(hh_sav.sum()) + gov_fixed + foreign_target + levy
    v = inv_share / np.maximum(pq, 1e-12)          # inv quantity per unit of nominal saving
    inv_q = (inv_share * investment_level / np.maximum(1.0 + reg.ptax, 1e-12)
             if fixed_investment else v * available0)
    F0 = hh_final + gov_q + inv_q

    link = (phi * (dsh * reg.arm_share)[None, :]
            / np.maximum(prices["dom_frac"], 1e-12))
    M = np.zeros((econ.n_act, C))
    for r in range(reg.n_reg):
        for c in range(C):
            i = reg.origin_index[r, c]
            M[:, c] += econ.msh[:, i] * link[r, c]
    io = reg.io / tfp[None, :]
    B = io @ M + reg.mq
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
    available = available0 + float(tax_per @ A)
    savings = float(pq @ inv_q) if fixed_investment else available
    ND = dsh * reg.arm_share * A
    Q_rc = link * A[None, :]
    D_rc = phi * ND[None, :]
    E_rc = prices["exp_frac"] * Q_rc
    M_c = prices["msh"] * reg.arm_share * A
    Q = np.zeros(econ.n_com)
    exports = np.zeros(econ.n_com)
    imports = np.zeros(econ.n_com)
    for r in range(reg.n_reg):
        for c in range(C):
            i = int(reg.origin_index[r, c])
            Q[i] = Q_rc[r, c]
            exports[i] = E_rc[r, c]
    imp0 = np.array([econ.imp0[reg.origin_index[r]] for r in range(reg.n_reg)])
    imp_w = imp0 / np.maximum(imp0.sum(axis=0), 1e-12)
    imp_w = np.where(imp0.sum(axis=0) > 0, imp_w, 0.0)
    for r in range(reg.n_reg):
        for c in range(C):
            imports[reg.origin_index[r, c]] = imp_w[r, c] * M_c[c]
    X = econ.msh @ Q
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
    inv_q_reg = inv_q * (reg.inv_dem_regional / max(float(inv_total.sum()), 1e-12))
    dest = hh_by + inter + gov_regional_q + inv_q_reg
    bilateral = phi[:, None, :] * dest[None, :, :]

    export_prices = (np.tile(prices["pe"], (reg.n_reg, 1)) if reg.cet
                     else prices["p_origin"][reg.origin_index])
    trade = float(prices["pm"] @ M_c - export_prices.ravel() @ exports)
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
        fac_dem=fac_dem, fac_by=fac_by, savings=savings, available=available, current=current,
        bilateral=bilateral, hh_nat=hh_q, gov_nat=gov_q, inv_nat=inv_q, ND=ND,
        hh_sav=hh_sav, gov_regional_q=gov_regional_q, inv_q_reg=inv_q_reg,
        migration=migration_results)


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
        household_saving=float(np.sum(state["hh_sav"])), forced_saving=state["levy"],
        foreign_saving=state["foreign_sav"],
        factor_pools=state["migration"] or None,
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
