"""Several regions inside one equilibrium.

Each destination has a CES across regions of origin, under the Armington the
one-region model already has. Households, activities and mapped institutions use
their calibrated origin cells; national government and investment share a final
channel. With a single origin the CES price is that origin's price, so a
one-region SAM written with a region label solves as the national model.

Delivered margins enter purchaser prices and quantities. Optional route margins
add shipment-proportional demand and a Cobb–Douglas global service pool. Under
``export_rule="cet"``, each origin splits output between home sales and exports
in response to the foreign price. A regional government closure and the JuMP
replica are not here. There is one national numeraire, and its market stays out
of the system so the Walras residual remains an independent check.

``bilateral`` reports domestic origin sales and ``bilateral_imports`` the
imported content on the same destination grid. Their sum reproduces the SAM's
regional buyer cells without routes; ``transport['vtwr']`` completes the
purchaser-value identity when route margins are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from .functions import (armington, ces_intensity, ces_price, cet, cet_domestic_price,
                        cobb_douglas_price)
from . import regional_jacobian
from .mobility import FactorMobility
from .solve import PeriodResult, factor_intensity, value_added_price
from .transport import TransportMargins

__all__ = ["Regional", "build_regional", "solve_regions"]

_PRICE_ITERS = 200
#: A start whose residual is the rounding a calibration leaves (1e-13 to 5e-13 on the
#: benchmark SAMs) is the solution; anything larger is refined, so a near miss never
#: stops at the closure tolerance.
_AT_START = 1e-12


@dataclass
class Regional:
    """The maps the interregional nest adds to a calibrated economy."""

    regions: tuple
    local_commodities: list
    #: Commodity index of origin ``(region, local)``.
    origin_index: np.ndarray
    #: Base-year share of each origin in domestic sales of the local commodity.
    origin_share: np.ndarray          # (n_reg, n_local)
    #: Origin shares by origin, destination channel and local commodity. The
    #: last channel combines national government and national investment.
    destination_share: np.ndarray     # (n_reg, n_reg + 1, n_local)
    import_share: np.ndarray          # imported-content attribution, same shape
    absorption0: np.ndarray           # (n_reg + 1, n_local)
    sigma: np.ndarray                 # (n_local,)
    dom_share: np.ndarray             # national Armington domestic share
    dom_share_dest: np.ndarray        # (n_reg + 1, n_local)
    arm_share: np.ndarray
    base_price: np.ndarray            # (n_reg + 1, n_local), purchaser prices
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
    gov_saving0: float = 0.0
    gov_revenue0: float = 0.0
    gov_savrate: float = 0.0
    transport: TransportMargins | None = None

    @property
    def n_reg(self) -> int:
        return len(self.regions)

    @property
    def n_local(self) -> int:
        return len(self.local_commodities)

    @property
    def n_dest(self) -> int:
        return self.n_reg + 1


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
            purchases = sam.block(list(econ.coms), [account]).ravel()
            demand[i] = purchases[origin_index].sum(axis=0)
    return accounts, demand


def _destination_shares(econ, sam, regions, origin_index, origin_share, ptax,
                        activity_region, household_region, government_accounts,
                        investment_accounts):
    n_reg, n_local = origin_index.shape
    use = np.zeros((n_reg, n_reg + 1, n_local))
    for d in range(n_reg):
        buyers = ([h for i, h in enumerate(econ.hhs) if household_region[i] == d]
                  + [a for j, a in enumerate(econ.acts) if activity_region[j] == d])
        for mapping in (government_accounts, investment_accounts):
            account = mapping.get(regions[d])
            if account is not None:
                buyers.append(account)
        for c in range(n_local):
            use[:, d, c] = sam.block(list(econ.coms), buyers).sum(axis=1)[origin_index[:, c]]
    national = [econ.layout.government, econ.layout.investment]
    for c in range(n_local):
        use[:, -1, c] = sam.block(list(econ.coms), national).sum(axis=1)[origin_index[:, c]]
    total = use.sum(axis=0)
    shares = np.divide(use, total[None, :, :], out=np.zeros_like(use),
                       where=total[None, :, :] > 0)
    for d in range(n_reg + 1):
        empty = total[d] <= 0
        shares[:, d, empty] = origin_share[:, empty]
    absorption0 = total / (1.0 + ptax)[None, :]
    return shares, absorption0, use


def _transport_calibration(econ, transport, regions, local, origin_index, use,
                           destination_share, absorption0, dom_share, arm_share, mq,
                           ptax, sigma, sigma_arm):
    K, C = use.shape[1:]
    yard = 1.0 + ptax
    if transport is None:
        # An origin account sells its region's output and the imports booked
        # through it. Splitting each cell by that origin's own home share gives
        # each destination its own import share; one national share misses the
        # base year whenever regions import different shares of a good.
        dom0 = econ.dom0[origin_index]
        through = dom0 + econ.imp0[origin_index]
        home = np.divide(dom0, through, out=np.ones_like(dom0), where=through > 0)
        arm = use / yard[None, None, :] * arm_share[None, None, :]
        domestic = arm * home[:, None, :]
        imported = arm - domestic
        dom_total, arm_total, imp_total = (domestic.sum(axis=0), arm.sum(axis=0),
                                           imported.sum(axis=0))
        destination_share = np.where(dom_total[None] > 0,
                                     domestic / np.maximum(dom_total[None], 1e-300),
                                     destination_share)
        import_share = np.where(imp_total[None] > 0,
                                imported / np.maximum(imp_total[None], 1e-300),
                                destination_share)
        dom_share_dest = np.where(arm_total > 0, dom_total / np.maximum(arm_total, 1e-300),
                                  dom_share[None, :])
        return (destination_share, import_share, absorption0, dom_share_dest,
                arm_share, mq, np.tile(yard, (K, 1)), None)
    if not isinstance(transport, TransportMargins):
        raise TypeError("transport must be a TransportMargins object")
    expected_destinations = tuple(regions) + ("national",)
    if (transport.regions != tuple(regions) or transport.commodities != tuple(local)
            or transport.destinations != expected_destinations):
        raise ValueError("transport labels must match regions, local commodities and "
                         f"destinations {expected_destinations}")
    raw = use / yard[None, None, :]
    imported = raw - transport.fob - transport.transport
    if imported.min() < -1e-8:
        raise ValueError("transport FOB plus margins exceeds a SAM absorption cell")
    if not np.allclose(transport.fob.sum(axis=1) + transport.supply,
                       econ.dom0[origin_index], rtol=1e-9, atol=1e-8):
        raise ValueError("transport FOB plus service supply does not reproduce domestic sales")
    if not np.allclose(imported.sum(axis=1), econ.imp0[origin_index],
                       rtol=1e-9, atol=1e-8):
        raise ValueError("SAM cells minus FOB and transport do not reproduce imports")
    margin_values = econ.mq * econ.A0[None, :]
    for r in range(len(regions)):
        for c in range(len(local)):
            i = origin_index[r, c]
            if not np.isclose(transport.supply[r, c], margin_values[i].sum(),
                              rtol=1e-9, atol=1e-8):
                raise ValueError(f"transport supply does not match the margin row for {econ.coms[i]}")
            if not np.isclose(transport.transport[r, :, c].sum(), margin_values[:, i].sum(),
                              rtol=1e-9, atol=1e-8):
                raise ValueError(f"route margins do not match the margin column for {econ.coms[i]}")

    destination_share = destination_share.copy()
    import_share = destination_share.copy()
    arm_share = arm_share.copy()
    mq = mq.copy()
    dom_share_dest = np.tile(dom_share, (K, 1))
    delivered = transport.fob + transport.transport
    delivered_total = delivered.sum(axis=0)
    raw_total = raw.sum(axis=0)
    routed = transport.transport.sum(axis=(0, 1)) > 0
    arm_share[routed] = 1.0
    mq[:, routed] = 0.0
    arm_price = np.ones((K, C))
    # A good without routes has no margins (the checks above tie margins to
    # routes), so the same calibration applies with a zero coefficient: its
    # FOB values, not its purchase shares, say how much each origin delivers.
    for c in np.where(routed | (raw_total.sum(axis=0) > 0))[0]:
        for d in range(K):
            total = delivered_total[d, c]
            if total <= 0:
                if not routed[c]:
                    if raw_total[d, c] > 0:  # bought only from abroad
                        dom_share_dest[d, c] = 0.0
                        import_share[:, d, c] = imported[:, d, c] / imported[:, d, c].sum()
                    continue
                raise ValueError(f"destination {d} has no absorption for transported "
                                 f"commodity {local[c]!r}")
            value_share = delivered[:, d, c] / total
            import_total = imported[:, d, c].sum()
            if import_total > 0:
                import_share[:, d, c] = imported[:, d, c] / import_total
            prices = 1.0 + transport.coefficient[:, d, c]
            theta = value_share * prices ** (sigma[c] - 1.0)
            destination_share[:, d, c] = theta / theta.sum()
            pd = float(ces_price(destination_share[:, d, c:c + 1], prices, sigma[c])[0])
            desired = delivered_total[d, c] / max(raw_total[d, c], 1e-12)
            domestic_weight = desired * pd ** (sigma_arm[c] - 1.0)
            dom_share_dest[d, c] = domestic_weight / (domestic_weight + 1.0 - desired)
            arm_price[d, c] = float(armington(
                np.array([dom_share_dest[d, c]]), np.array([pd]), np.array([1.0]),
                np.array([sigma_arm[c]]))[0][0])
    basic_price = np.ones((K, C))
    for _ in range(_PRICE_ITERS):
        new = arm_price * arm_share + basic_price @ mq
        if np.max(np.abs(new - basic_price)) < 1e-14:
            basic_price = new
            break
        basic_price = new
    else:
        raise RuntimeError("base regional margin prices did not converge")
    absorption0 = raw_total / np.maximum(basic_price, 1e-12)
    return (destination_share, import_share, absorption0, dom_share_dest, arm_share, mq,
            basic_price * yard, transport)


def build_regional(econ, sam, *, migration_pools=None, transport=None) -> Regional:
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

    destination_share, absorption0, use = _destination_shares(
        econ, sam, regions, origin_index, origin_share, ptax, activity_region,
        household_region, government_accounts, investment_accounts)
    (destination_share, import_share, absorption0, dom_share_dest, nat_arm, nat_mq,
     base_price, transport) = _transport_calibration(
        econ, transport, regions, local, origin_index, use, destination_share,
        absorption0, dom_share, nat_arm, nat_mq, ptax, sigma, sigma_arm)
    # Activities buy at their own region's purchaser price. With route margins
    # that base price is not the tax-inclusive one, and the coefficient is in
    # units of what the activity pays.
    basic_price = base_price / (1.0 + ptax)[None, :]
    nat_io = nat_io / basic_price[activity_region].T
    migration = _build_migration(spec, econ, regions, migration_pools)
    dtax = econ.zeros_if_none("hh_dtax", econ.n_hh)
    direct = float((np.asarray(econ.hh_gross0, float) * dtax).sum())
    transfers = float(econ.zeros_if_none("hh_gov_transfer", econ.n_hh).sum())
    factor_gov = float((econ.fac_supply * (econ.zeros_if_none("fac_gov_share", econ.n_fac)
                                           + fac_gov_regional.sum(axis=0))).sum())
    revenue0 = (float(ptax @ econ.A0[origin_index].sum(axis=0))
                + float(econ.atax @ econ.X0)
                + direct + factor_gov)
    consumption0 = float(nat_gov.sum() + gov_regional.sum())
    saving0 = revenue0 - transfers - consumption0
    net_revenue = revenue0 - transfers
    savrate = saving0 / net_revenue if net_revenue > 0 else float(econ.gov_savrate)

    return Regional(
        regions=regions, local_commodities=local, origin_index=origin_index,
        origin_share=origin_share, destination_share=destination_share,
        import_share=import_share, absorption0=absorption0, sigma=sigma, dom_share=dom_share,
        dom_share_dest=dom_share_dest, arm_share=nat_arm, base_price=base_price,
        ptax=ptax, sigma_arm=sigma_arm,
        hh_budget=nat_budget, gov_dem=nat_gov, inv_dem=nat_inv, io=nat_io,
        exp_share=exp_share, sigma_cet=np.repeat(sigma_cet[None, :], n_reg, axis=0),
        mq=nat_mq, cet=spec.export_rule == "cet", activity_region=activity_region,
        household_region=household_region, government_accounts=government_accounts,
        investment_accounts=investment_accounts,
        gov_dem_regional=gov_regional, inv_dem_regional=inv_regional,
        fac_gov_regional=fac_gov_regional, migration=migration,
        gov_saving0=saving0, gov_revenue0=revenue0, gov_savrate=savrate,
        transport=transport)


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
    target = closure.government_target
    adjusts = closure.government_adjusts
    tax_lever = target is not None and adjusts in ("direct_tax", "product_tax")
    i_tax = i_inv + int(fixed_investment)
    n_unknown = i_tax + int(tax_lever)
    nu = closure.numeraire
    target_base = reg.gov_saving0 if target == "saving" else reg.gov_revenue0
    target_value = (target_base if closure.government_target_value is None
                    else float(closure.government_target_value)) * nu
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
    ptax_set = reg.ptax if scenario.product_tax is None else np.asarray(
        scenario.product_tax, dtype=float)
    if ptax_set.shape != (reg.n_local,):
        raise ValueError("product_tax in a regional model is one rate per local "
                         f"commodity ({reg.local_commodities}), got shape {ptax_set.shape}")
    investment_level = ((reg.inv_dem + reg.inv_dem_regional.sum(axis=0)).sum()
                        if scenario.investment_level is None else float(scenario.investment_level))
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
        tax_scale = float(np.exp(z[i_tax])) if tax_lever else 1.0
        dtax_eff = dtax * (tax_scale if adjusts == "direct_tax" and tax_lever else 1.0)
        ptax_eff = ptax_set * (tax_scale if adjusts == "product_tax" and tax_lever else 1.0)
        prices = _prices(econ, reg, w, er, world, export_world, tfp, nu, ptax_eff)
        qty = _quantities(econ, reg, w, supply, facshare, fac_gov, fac_row, dtax_eff,
                          transfers - levy * econ.pop_share, gov_tr, remit,
                          foreign_target, tfp, prices, nu,
                          investment_level=investment_level, levy=levy,
                          fixed_investment=fixed_investment, migration=reg.migration,
                          ptax=ptax_eff, closure=closure, target_value=target_value)
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
        if tax_lever:
            hit = qty["government"]["saving" if target == "saving" else "revenue"]
            out.append((hit - target_value) / max(abs(target_value), 1.0))
        state.update(prices)
        state.update(qty)
        state.update(w=w, er=er, gaps=gaps, walras=float(gaps[num]),
                     levy=levy, foreign_sav=foreign_target, tax_scale=tax_scale,
                     at=z.copy(), blocks=(prices, qty))
        return np.asarray(out, dtype=float)

    def jacobian(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if "at" not in state or not np.array_equal(state["at"], z):
            residuals(z)
        ctx = dict(n_unknown=n_unknown, others=others, i_er=i_er, i_inv=i_inv,
                   fixed_investment=fixed_investment, closure=closure, nu=nu,
                   w=state["w"], er=state["er"], levy=state["levy"], tfp=tfp,
                   supply=supply, world=world, export_world=export_world, ptax=ptax_set,
                   facshare=facshare, fac_gov=fac_gov, fac_row=fac_row, dtax=dtax,
                   gov_tr=gov_tr, foreign_target=foreign_target)
        return regional_jacobian.jacobian(econ, ctx, *state["blocks"])

    exact = jacobian if regional_jacobian.supported(closure) else None
    z_hat, residual, label, message = _solve(residuals, n_unknown, closure.tolerance,
                                             warm_start, raise_on_fail, jac=exact)
    residuals(z_hat)
    return _pack(econ, reg, closure, scenario, state, z_hat, residual, label, message,
                 closure.tolerance)


def _refuse_unsupported(econ, closure, scenario, reg: Regional):
    if closure.labour != "full_employment":
        raise ValueError(
            "regional factors clear where they are supplied. "
            f"closure.labour={closure.labour!r} would move a wage without a "
            "migration equation, and the two are different closures.")
    if closure.investment not in ("savings_driven", "fixed_real"):
        raise ValueError(f"unsupported regional investment closure {closure.investment!r}")
    if closure.government not in ("real_fixed", "nominal_fixed", "saving_rate"):
        raise ValueError(f"unsupported regional government closure {closure.government!r}")
    if closure.government_adjusts == "direct_tax" and closure.government_target is not None:
        dtax = econ.zeros_if_none("hh_dtax", econ.n_hh)
        if not np.any(dtax != 0):
            raise ValueError("government_adjusts='direct_tax', but no household pays direct tax")
    if closure.government_adjusts == "product_tax" and closure.government_target is not None:
        ptax = reg.ptax if scenario.product_tax is None else np.asarray(scenario.product_tax)
        if not np.any(ptax != 0):
            raise ValueError("government_adjusts='product_tax', but every product tax rate is zero")
    if closure.foreign != "fixed_saving":
        raise ValueError(
            "the exchange rate clears one national current account. "
            f"closure.foreign={closure.foreign!r} is not in this slice.")
    blocked = {
        "consumer_wedge": scenario.consumer_wedge,
        "government_wedge": scenario.government_wedge,
        "product_tax_firm": scenario.product_tax_firm,
        "investment_mix": scenario.investment_mix,
        "foreign_factor_share": scenario.foreign_factor_share,
        "tariff": scenario.tariff,
        "output_tax": scenario.output_tax,
        "carbon_price": scenario.carbon_price,
        "pool_supply": scenario.pool_supply,
        "labour_force": scenario.labour_force,
    }
    named = [k for k, v in blocked.items() if v is not None]
    if scenario.government_absorbs != 0.0:
        named.append("government_absorbs")
    if scenario.capacity:
        named.append("capacity")
    if scenario.import_quota:
        named.append("import_quota")
    if scenario.price_ceiling:
        named.append("price_ceiling")
    if scenario.price_floor:
        named.append("price_floor")
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


def _prices(econ, reg: Regional, w, er, world, export_world, tfp, nu, ptax):
    """CET origin sales and a delivered Armington composite in each destination."""
    io = reg.io / tfp[None, :]
    vunit = econ.vunit / tfp
    atax = econ.atax * nu
    K, C = reg.n_dest, reg.n_local
    pq = reg.base_price.copy() if reg.transport is not None else np.ones((K, C))
    p_origin = np.ones(econ.n_com)
    pa = np.ones(econ.n_act)
    pd_dest = np.ones((K, C))
    arm_price = np.ones((K, C))
    phi = reg.destination_share.copy()
    dsh = reg.dom_share_dest.copy()
    msh = 1.0 - dsh
    pe = er * export_world
    pm = er * world
    shares = reg.destination_share.reshape(reg.n_reg, K * C)
    sigma_col = np.tile(reg.sigma, K)
    sigma_arm = np.broadcast_to(reg.sigma_arm, (K, C))
    for _ in range(_PRICE_ITERS):
        pva = value_added_price(econ, w)
        pa = (io.T * pq[reg.activity_region]).sum(axis=1) + pva * vunit + atax
        p_origin = econ.msh.T @ pa
        p_rc = p_origin[reg.origin_index]
        if reg.cet:
            p_rc = cet_domestic_price(reg.exp_share, p_rc, pe[None, :], reg.sigma_cet)
        if reg.transport is None:
            transport_price = 0.0
            p_deliver = np.tile(p_rc[:, None, :], (1, K, 1))
        else:
            # Services reach the pool as home sales, at the home-sales price.
            transport_price = reg.transport.pool_price(p_rc)
            p_deliver = p_rc[:, None, :] + reg.transport.coefficient * transport_price
        # Every destination and good at once: one CES column per (d, c), in
        # the row-major order of the (K, C) grid.
        prices = p_deliver.reshape(reg.n_reg, K * C)
        pd_dest = ces_price(shares, prices, sigma_col).reshape(K, C)
        phi = ces_intensity(shares, prices, pd_dest.ravel(), sigma_col).reshape(phi.shape)
        arm_price, dsh, msh = armington(reg.dom_share_dest, pd_dest,
                                        np.broadcast_to(pm, (K, C)), sigma_arm)
        pbase = arm_price * reg.arm_share + pq @ reg.mq
        new = pbase * (1.0 + ptax)
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
    return dict(pa=pa, p_origin=p_origin, pd_dest=pd_dest, phi=phi, pq=pq,
                pbase=pq / np.maximum(1.0 + ptax, 1e-12), dsh=dsh, msh=msh,
                pm=pm, pe=pe, dom_frac=dom_frac, exp_frac=exp_frac,
                p_ship=p_rc, p_deliver=p_deliver, transport_price=transport_price)


def _quantities(econ, reg, w, supply, facshare, fac_gov, fac_row, dtax, transfers,
                gov_tr, remit, foreign_target, tfp, prices, nu, *,
                investment_level, levy, fixed_investment, migration=None,
                ptax=None, closure=None, target_value=0.0):
    """Solve destination absorption and origin sales in one linear system."""
    K, C = reg.n_dest, reg.n_local
    pq, phi, dsh = prices["pq"], prices["phi"], prices["dsh"]
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
    hh_q = reg.hh_budget * spend[:, None] / np.maximum(pq[reg.household_region], 1e-12)
    hh_by = np.zeros((K, C))
    for h in range(econ.n_hh):
        hh_by[reg.household_region[h]] += hh_q[h]

    gov_values = np.zeros((K, C))
    gov_values[-1] = reg.gov_dem
    gov_values[:reg.n_reg] = reg.gov_dem_regional
    gov_base_q = gov_values / np.maximum(reg.base_price, 1e-12)
    gov_total = float(gov_values.sum())
    gov_mix = gov_values / gov_total if gov_total > 0 else gov_values
    income_excl = (float((fac_inc * fac_gov).sum())
                   + float((fac_inc * reg.fac_gov_regional).sum())
                   + float((hh_income * dtax).sum()) - float(gov_tr.sum()))
    dynamic_gov = (closure.government == "saving_rate"
                   or (closure.government_target == "saving"
                       and closure.government_adjusts == "spending"))
    if dynamic_gov:
        gov_channel_q = np.zeros((K, C))
        available0 = 0.0
        gov_q = np.zeros(C)
    else:
        if closure.government == "nominal_fixed":
            gov_channel_q = gov_values * nu / np.maximum(pq, 1e-12)
        else:
            gov_channel_q = gov_base_q
        gov_q = gov_channel_q.sum(axis=0)
        gov_spend = float((pq * gov_channel_q).sum())
        available0 = float(hh_sav.sum()) + income_excl - gov_spend + foreign_target + levy
    gov_regional_q = gov_channel_q[:reg.n_reg]
    inv_total = reg.inv_dem + reg.inv_dem_regional.sum(axis=0)
    inv_channel = np.zeros((K, C))
    inv_channel[-1] = reg.inv_dem
    inv_channel[:reg.n_reg] = reg.inv_dem_regional
    inv_base_q = inv_channel / np.maximum(reg.base_price, 1e-12)
    inv_base_cost = float(inv_total.sum())
    # Commodity value shares are fixed; within a commodity the channels keep
    # their base-year quantities. Weighting channels by base values instead
    # misses the base year whenever destinations pay different base prices.
    inv_share_channel = inv_base_q / inv_base_cost if inv_base_cost > 0 else inv_base_q
    inv_price = np.divide((inv_base_q * pq).sum(axis=0), inv_total,
                          out=np.ones(C), where=inv_total > 0)
    v_channel = inv_share_channel / np.maximum(inv_price[None, :], 1e-12)
    inv_channel_q = (inv_base_q * investment_level / inv_base_cost
                     if fixed_investment and inv_base_cost > 0
                     else v_channel * available0)
    inv_q = inv_channel_q.sum(axis=0)
    F0 = hh_by + gov_channel_q + inv_channel_q

    link = phi * (dsh * reg.arm_share)[None, :, :]
    if reg.transport is None:
        coefficient = np.zeros_like(link)
        service_share = np.zeros((reg.n_reg, C))
    else:
        coefficient = reg.transport.coefficient
        service_prices = prices["p_ship"]
        service_share = (reg.transport.supply / reg.transport.supply.sum()
                         * prices["transport_price"] / service_prices)
    transport_link = (coefficient * link).sum(axis=0)
    # Service supply to the pool is part of home sales, so it is grossed up by
    # the home share of output like the bilateral sales are.
    output_link = ((link + service_share[:, None, :] * transport_link[None, :, :])
                   / np.maximum(prices["dom_frac"][:, None, :], 1e-12))
    # Output of each activity per unit of absorption in (d, c), through the
    # origins that supply it.
    M = np.einsum("arc,rdc->adc", econ.msh[:, reg.origin_index], output_link)
    io = reg.io / tfp[None, :]
    IO = np.zeros((K, C, econ.n_act))
    IO[reg.activity_region, :, np.arange(econ.n_act)] = io.T
    B = IO.reshape(K * C, econ.n_act) @ M.reshape(econ.n_act, K * C)
    B += np.kron(np.eye(K), reg.mq)
    ptax = reg.ptax if ptax is None else ptax
    tax_per = prices["pbase"] * ptax + (M.reshape(econ.n_act, K * C).T
                                         @ (econ.atax * nu)).reshape(K, C)
    eye = np.eye(K * C)
    try:
        L = np.linalg.solve(eye - B, eye)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "the regional quantity system is singular. A commodity is used in "
            "its own production with a coefficient of one or more.") from exc
    if dynamic_gov:
        free = ["gov"] + ([] if fixed_investment else ["sav"])

        def flows(u):
            gov_budget = float(u[0])
            inv_budget = float(u[-1]) if not fixed_investment else 0.0
            gov_q_now = gov_mix * gov_budget / np.maximum(pq, 1e-12)
            inv_q_now = (inv_base_q * investment_level / inv_base_cost
                         if fixed_investment and inv_base_cost > 0
                         else v_channel * inv_budget)
            A_now = (L @ (hh_by + gov_q_now + inv_q_now).ravel()).reshape(K, C)
            income = income_excl + float((tax_per * A_now).sum())
            spend_now = float((pq * gov_q_now).sum())
            avail = (float(hh_sav.sum()) + levy + foreign_target + income - spend_now)
            return gov_q_now, inv_q_now, A_now, income, avail

        def gap(u):
            _, _, _, income, avail = flows(u)
            out = []
            if closure.government == "saving_rate":
                out.append(u[0] - (1.0 - reg.gov_savrate) * income)
            else:
                out.append(u[0] - (income - target_value))
            if not fixed_investment:
                out.append(u[-1] - avail)
            return np.asarray(out)

        r0 = gap(np.zeros(len(free)))
        J = np.empty((len(free), len(free)))
        scale = max(abs(foreign_target), 1.0)
        for k in range(len(free)):
            step = np.zeros(len(free))
            step[k] = scale
            J[:, k] = (gap(step) - r0) / scale
        u = np.linalg.solve(J, -r0)
        gov_channel_q, inv_channel_q, A, _, available = flows(u)
        gov_regional_q = gov_channel_q[:reg.n_reg]
        gov_q = gov_channel_q.sum(axis=0)
        gov_spend = float((pq * gov_channel_q).sum())
        inv_q = inv_channel_q.sum(axis=0)
        savings = float((pq * inv_channel_q).sum()) if fixed_investment else available
    else:
        # Tax receipts fund extra investment only in the savings-driven closure.
        # With fixed quantities they change the financing levy, not demand.
        leak = (np.zeros_like(eye) if fixed_investment else
                L @ np.outer(v_channel.ravel(), tax_per.ravel()))
        try:
            A = np.linalg.solve(eye - leak, L @ F0.ravel()).reshape(K, C)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "product-tax revenue feeds investment which feeds the tax base, and "
                "that loop does not have a finite sum.") from exc
        available = available0 + float((tax_per * A).sum())
        if not fixed_investment:
            inv_channel_q = v_channel * available
            inv_q = inv_channel_q.sum(axis=0)
        savings = float((pq * inv_channel_q).sum()) if fixed_investment else available
    D_rc = phi * (dsh * reg.arm_share * A)[None, :, :]
    Q_rc = ((D_rc.sum(axis=1) + service_share * (coefficient * D_rc).sum())
            / np.maximum(prices["dom_frac"], 1e-12))
    E_rc = prices["exp_frac"] * Q_rc
    M_c = prices["msh"] * reg.arm_share * A
    origin_mix = (phi if reg.transport is None else
                  phi * dsh[None, :, :] + reg.import_share * prices["msh"][None, :, :])
    Q = np.zeros(econ.n_com)
    exports = np.zeros(econ.n_com)
    imports = np.zeros(econ.n_com)
    Q[reg.origin_index] = Q_rc
    exports[reg.origin_index] = E_rc
    imports[reg.origin_index] = (reg.import_share * M_c[None, :, :]).sum(axis=1)
    X = econ.msh @ Q
    fi = factor_intensity(econ, w, vunit=econ.vunit / tfp, aunit=econ.aunit / tfp[None, :])
    fac_by = fi * ((econ.vunit / tfp) * X)[None, :]
    fac_dem = fac_by.sum(axis=1)

    export_prices = (np.tile(prices["pe"], (reg.n_reg, 1)) if reg.cet
                     else prices["p_origin"][reg.origin_index])
    trade = float(prices["pm"] @ M_c.sum(axis=0) - export_prices.ravel() @ exports)
    current = trade + float((fac_inc * fac_row).sum()) - float(remit.sum())
    hh_origin = np.zeros((econ.n_hh, econ.n_com))
    gov_origin = np.zeros(econ.n_com)
    inv_origin = np.zeros(econ.n_com)
    hh_origin[:, reg.origin_index] = (origin_mix[:, reg.household_region, :]
                                      * hh_q[None, :, :]).transpose(1, 0, 2)
    gov_origin[reg.origin_index] = (origin_mix[:, -1, :] * gov_channel_q[-1][None, :]
                                    + (origin_mix[:, :reg.n_reg, :]
                                       * gov_regional_q[None, :, :]).sum(axis=1))
    inv_origin[reg.origin_index] = (origin_mix * inv_channel_q[None, :, :]).sum(axis=1)
    government_saving = available - float(hh_sav.sum()) - levy - foreign_target
    gov_account = dict(
        product_tax=float((prices["pbase"] * ptax * A).sum()),
        activity_tax=float((econ.atax * nu) @ X),
        carbon_production=0.0, carbon_households=0.0, tariff=0.0, output_tax=0.0,
        direct_tax=float((hh_income * dtax).sum()),
        factor_income=float((fac_inc * fac_gov).sum()
                            + (fac_inc * reg.fac_gov_regional).sum()),
        wedge=0.0, transfers=float(gov_tr.sum()), consumption=gov_spend, absorbed=0.0)
    gov_account["revenue"] = sum(gov_account[k] for k in (
        "product_tax", "activity_tax", "carbon_production", "carbon_households",
        "tariff", "output_tax", "direct_tax", "factor_income", "wedge"))
    gov_account["saving"] = (gov_account["revenue"] - gov_account["transfers"]
                             - gov_account["consumption"] - gov_account["absorbed"])
    pe_full = np.zeros(econ.n_com)
    pe_full[reg.origin_index] = export_prices
    domestic_sales = np.zeros(econ.n_com)
    domestic_sales[reg.origin_index] = D_rc.sum(axis=1)
    transport_result = None
    if reg.transport is not None:
        transport_result = reg.transport.evaluate(
            D_rc, np.tile(prices["p_ship"][:, None, :], (1, K, 1)), prices["p_ship"])
    return dict(
        A=A.sum(axis=0), A_dest=A, Q=Q, X=X, D=D_rc, exports=exports, imports=imports,
        hh_q=hh_origin, hh_income=hh_income, spend=spend, gov_q=gov_origin, inv_q=inv_origin,
        fac_dem=fac_dem, fac_by=fac_by, savings=savings, available=available, current=current,
        bilateral=D_rc[:, :reg.n_reg, :],
        bilateral_imports=(reg.import_share * M_c[None, :, :])[:, :reg.n_reg, :],
        hh_nat=hh_q, gov_nat=gov_q, inv_nat=inv_q,
        ND=(dsh * reg.arm_share * A).sum(axis=0), hh_sav=hh_sav,
        government_saving=government_saving,
        gov_regional_q=gov_regional_q, inv_q_reg=inv_channel_q[:reg.n_reg],
        gov_channel_q=gov_channel_q, inv_channel_q=inv_channel_q,
        government=gov_account, pe=pe_full, domestic_sales=domestic_sales,
        p_domestic=prices["p_ship"], hh_direct_tax=hh_income * dtax,
        row_fac=float((fac_inc * fac_row).sum()),
        migration=migration_results, transport=transport_result)


def _solve(residuals, n_unknown, tolerance, warm_start, raise_on_fail, jac=None):
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
            # A start that already solves the system (the base year, or a warm
            # start on an unchanged period) is the answer; hybr would spend a
            # Jacobian per unknown on it.
            first = np.asarray(residuals(start), dtype=float)
            if first.size == 0 or float(np.max(np.abs(first))) < min(tolerance, _AT_START):
                best = (float(np.max(np.abs(first))) if first.size else 0.0,
                        np.asarray(start, dtype=float), label, "the start point solves the system")
                break
            sol = root(residuals, start, method=method, jac=jac,
                       options={"xtol": 1e-14} if method == "hybr"
                       else {"xtol": 1e-14, "ftol": 1e-14})
        except (ValueError, RuntimeError):
            continue
        z = np.asarray(sol.x, dtype=float)
        res = float(np.max(np.abs(residuals(z))))
        if jac is not None:
            z, res = _polish(residuals, jac, z, res)
        if best is None or res < best[0]:
            best = (res, z, label, str(sol.message))
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


def _polish(residuals, jac, z, res, steps=2):
    """Newton steps with the exact Jacobian, kept only while they lower the residual.

    hybr stops on the size of its step, which can leave a residual a hundred
    times above rounding. With the exact Jacobian one more step costs one
    evaluation and lands on rounding.
    """
    for _ in range(steps):
        try:
            trial = z - np.linalg.solve(jac(z), residuals(z))
            new = float(np.max(np.abs(residuals(trial))))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            break
        if not new < res:
            break
        z, res = trial, new
    residuals(z)
    return z, res


def _pack(econ, reg, closure, scenario, state, z, residual, label, message, tolerance):
    weight = reg.absorption0 / np.maximum(reg.absorption0.sum(axis=0), 1e-12)[None, :]
    pq_nat = (weight * state["pq"]).sum(axis=0)
    pbase_nat = (weight * state["pbase"]).sum(axis=0)
    pq_full = np.zeros(econ.n_com)
    pbase_full = np.zeros(econ.n_com)
    world_export = np.zeros(econ.n_com)
    world_import = np.zeros(econ.n_com)
    for r in range(reg.n_reg):
        for c in range(reg.n_local):
            i = int(reg.origin_index[r, c])
            pq_full[i] = pq_nat[c]
            pbase_full[i] = pbase_nat[c]
            # pe is already expanded by origin; pm still has one price per
            # local commodity. Both are in domestic currency inside the solve.
            world_export[i] = state["pe"][i] / state["er"]
            world_import[i] = state["pm"][c] / state["er"]
    # The exact Cobb–Douglas index of the destination purchaser prices, the
    # same object the national solver reports: a one-region model gives the
    # national level, not only the national change.
    hh_index = cobb_douglas_price(reg.hh_budget.T, state["pq"][reg.household_region].T)
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
        foreign_saving=state["foreign_sav"], tax_scale=state["tax_scale"],
        factor_pools=state["migration"] or None,
        pe=state["pe"], hh_direct_tax=state["hh_direct_tax"],
        factor_income_abroad=state["row_fac"],
        world_export_price=world_export, world_import_price=world_import,
        domestic_sales=state["domestic_sales"], p_domestic=state["p_domestic"],
        government=state["government"],
        factor_prices_by_activity=np.broadcast_to(state["w"][:, None], state["fac_by"].shape),
        real=dict(
            hh_real=float((reg.base_price[reg.household_region] * state["hh_nat"]).sum()),
            gov_real=float((reg.base_price * state["gov_channel_q"]).sum()),
            inv_real=float((reg.base_price * state["inv_channel_q"]).sum()),
            hh_real_income=np.where(hh_index > 0, state["spend"] / hh_index, 0.0),
            hh_price_index=hh_index,
            national_absorption=state["A"],
            national_price=pq_nat,
            # Final demand by destination channel, at the prices each one pays.
            destination_price=state["pq"],
            hh_destination_quantity=state["hh_nat"],
            gov_destination_quantity=state["gov_channel_q"],
            inv_destination_quantity=state["inv_channel_q"],
            government_saving=state["government_saving"],
        ),
        gaps={f: float(g) for f, g in zip(econ.factors, state["gaps"])},
        walras=state["walras"], residual=residual, converged=ok, attempt=label,
        message=message, bilateral=state["bilateral"],
        bilateral_imports=state["bilateral_imports"], transport=state["transport"])


def _absorb(reg, state):
    """Regional domestic sales plus the imports attributed to that origin."""
    out = np.zeros(state["Q"].shape)
    for r in range(reg.n_reg):
        for c in range(reg.n_local):
            i = int(reg.origin_index[r, c])
            out[i] = state["D"][r, :, c].sum() + state["imports"][i]
    return out
