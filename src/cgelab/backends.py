"""Solve the period equilibrium as one simultaneous system, in Pyomo.

``solve_period`` solves by nesting: a Newton step on a few outer unknowns, and
inside it the delivered-price fixed point, two sums in closed form and a
Leontief inversion. Here nothing is nested. Every block is a constraint of one
system, the same system ``to_jump`` writes for JuMP, read from the same data
(:func:`cgelab.export.replica_data`), and an external solver finds the point
where all of them hold. ``solve_period(..., backend="ipopt")`` then maps that
point onto its own unknowns and evaluates its own residual there, so a
solution is accepted only if cgelab's equations hold at it as well.

With ``backend="path"`` the same system also carries the complementarity
conditions solve_period writes with the Fischer–Burmeister function: least-cost
producers, capacity limits, import quotas, price bounds and the wage floor. Each
is written with ``pyomo.mpec`` as a condition paired with the variable it holds
at zero, and PATH solves the mixed complementarity problem as a whole.

The solver programs are not Python packages. IPOPT and PATH are found on
``PATH`` or in the modules of ``amplpy`` (``python -m amplpy.modules install
ipopt path``). Without a licence string PATH stops above 300 variables; it
reads the free one from the environment variable ``PATH_LICENSE_STRING``.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

__all__ = ["solve_replica", "BACKENDS"]

#: Backends ``solve_period`` accepts besides its own ``"scipy"``.
BACKENDS = ("ipopt", "path")

#: The program each backend runs: amplpy's PATH module installs ``path``, other
#: distributions ``pathampl``.
_PROGRAMS = {"ipopt": ("ipopt",), "path": ("pathampl", "path")}


def _pyomo(backend="ipopt"):
    try:
        import pyomo.environ as pe
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(f"backend={backend!r} needs Pyomo: pip install 'cgelab[pyomo]'") from err
    return pe


def solver_executable(name: str) -> str:
    """Path of a solver's program: on ``PATH``, or among the modules of ``amplpy``."""
    try:
        from amplpy import modules
        module_path = modules.path()
    except Exception:  # amplpy absent, or no modules installed
        module_path = None
    for program in _PROGRAMS.get(name, (name,)):
        found = shutil.which(program) if program != "path" else None
        if not found and module_path:
            found = shutil.which(program, path=module_path)
        if found:
            return found
    raise RuntimeError(
        f"no {name!r} executable on PATH or among amplpy's modules. Install it, for "
        f"instance with: pip install amplpy && python -m amplpy.modules install {name}")


def solve_replica(econ, closure, scenario, backend: str = "ipopt") -> dict:
    """Solve the simultaneous replica and return the values ``solve_period`` maps onto its unknowns."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be 'scipy' or one of {BACKENDS}, got {backend!r}")
    from .export import replica_data
    mcp = backend == "path"
    d = replica_data(econ, closure, scenario, who=f"backend={backend!r}", complementarity=mcp)
    pe = _pyomo(backend)
    if mcp:
        opt = pe.SolverFactory("path", executable=solver_executable("path"))
        # The AMPL driver's names for PATH's convergence tolerance and limits.
        opt.options.update({"convtol": 1e-12, "majiterlim": 500, "cumuliterlim": 100000,
                            "outlev": 0})
        if os.environ.get("PATH_LICENSE_STRING"):
            opt.options["license"] = os.environ["PATH_LICENSE_STRING"]
    else:
        opt = pe.SolverFactory("ipopt", executable=solver_executable("ipopt"))
        # The flag is not the verdict: with a constant objective the problem is pure
        # feasibility. cgelab's own residual at the point decides.
        opt.options.update({"tol": 1e-12, "constr_viol_tol": 1e-12, "acceptable_tol": 1e-10,
                            "max_iter": 3000, "nlp_scaling_method": "gradient-based",
                            "nlp_scaling_max_gradient": 100.0, "print_level": 0})
    name = "PATH" if mcp else "IPOPT"
    # From the base year first; if the solver does not certify a point there, from
    # the displaced start of the JuMP replica, as the scipy solver tries a second start.
    for start in ("base", "displaced"):
        m, h = build_model(d, start=start)
        try:
            results = opt.solve(m, tee=False, load_solutions=False)
        except Exception as err:  # the program failed before writing a solution
            status, results = f"error: {err}", None
            continue
        status = str(results.solver.termination_condition)
        if status in ("optimal", "locallyOptimal"):
            break
    if results is None or len(results.solution) == 0:
        hint = (" Above 300 variables PATH needs a licence string in PATH_LICENSE_STRING: "
                "https://pages.cs.wisc.edu/~ferris/path/LICENSE" if mcp else "")
        raise RuntimeError(f"backend={backend!r}: {name} returned no point ({status}).{hint}")
    m.solutions.load_from(results)
    v = pe.value
    out = dict(
        status=f"{status} from the {start} start",
        w=np.array([v(h["w"][f]) for f in range(d["nf"])]),
        w_act={k: v(x) for k, x in h["wm"].items()},
        er=float(v(h["er"])), fsav=float(v(h["fsav"])),
        pwe=np.array([v(h["pwe"][c]) for c in range(d["nc"])]),
        pwm=np.array([v(h["pwm"][c]) for c in range(d["nc"])]),
        tscale=float(v(h["tscale"])), levy=float(v(h["levy"])),
        unemployment=[float(v(u)) for u in h["U"]],
        n_variables=sum(1 for _ in m.component_data_objects(pe.Var)),
    )
    if mcp:
        out.update(
            price={c: float(v(h["pd"][c])) for c in {g["commodity"] for g in d["least_cost"]}},
            share={k: float(v(x)) for k, x in h["share"].items()},
            rent={j: float(v(x)) for j, x in h["rent"].items()},
            premium={c: float(v(x)) for c, x in h["premium"].items()},
            subsidy={c: float(v(x)) for c, x in h["subsidy"].items()},
            bought={c: float(v(x)) for c, x in h["bought"].items()},
        )
    return out


def build_model(d: dict, start: str = "base"):
    """The Pyomo model of the period equilibrium, equation for equation the JuMP template's.

    ``start`` is ``"base"``, the base year with prices scaled by the numeraire,
    or ``"displaced"``, the JuMP replica's: non-numeraire factor prices at 0.85
    and the exchange rate at 1.15 of the numeraire's. Returns the model and a
    dictionary of the variables ``solve_period`` reads.
    The numeraire factor's market is left out, as in cgelab, so that its gap
    stays a check by Walras' law and not an identity.
    """
    pe = _pyomo()
    log = pe.log
    m = pe.ConcreteModel()
    m.eq = pe.ConstraintList()
    add = m.eq.add

    na, nc, nh, nf = d["na"], d["nc"], d["nh"], d["nf"]
    NUM = d["numeraire_index"] - 1
    nu = float(d["numeraire"])
    LAB = int(d["labour_index"]) - 1
    LABOUR = d["labour"]
    slack = LABOUR in ("fixed_real_wage", "wage_curve")

    def a(key):
        return np.asarray(d[key], dtype=float)

    io, msh, theta, aunit, mq = a("io"), a("msh"), a("theta"), a("aunit"), a("mq")
    vunit, leo = a("vunit"), [bool(b) for b in d["leontief"]]
    sva, sarm = a("sigma_va"), a("sigma_arm")
    X0, Q0, A0 = a("X0"), a("Q0"), a("A0")
    domsh, armsh, expsh = a("dom_share"), a("arm_share"), a("exp_share")
    wp, wpe, scet = a("world_price"), a("export_price"), a("sigma_cet")
    CET = bool(d["cet"])
    eta_x, psi_m = a("export_demand"), a("import_supply")
    E0, M0 = a("exp0"), a("imp0")
    dtax, gov_tr, remit = a("hh_dtax"), a("gov_transfer"), a("remit_net")
    gsh, rsh = a("fac_gov_share"), a("fac_row_share")
    ptax_a, ptax_f, atax = a("ptax_a"), a("ptax_f"), a("atax")
    tm, tm0, tauz, tauz0 = a("tm"), a("tm0"), a("tauz"), a("tauz0")
    hh_budget, beta_les, gamma = a("hh_budget"), a("beta_les"), a("hh_committed")
    coef, Wt = a("demand_coef"), a("Wt")
    g_cap, g_gov = a("g_cap"), a("g_gov")
    gamma_sum, gam_W, gam_Wg = a("gamma_sum"), a("gam_W"), a("gam_Wg")
    savrate, facshare, hh_direct = a("hh_savrate"), a("facshare"), a("hh_direct")
    transfers, les, popsh = a("transfers"), bool(d["les"]), a("pop_share")
    W, Wg, alpha = a("wedge"), a("wedge_gov"), a("recycle_share")
    gov_absorbs = float(d["gov_absorbs"])
    supply, inv_dem, inv_q_fix, inv_w = a("fac_supply"), a("inv_dem"), a("inv_q_fixed"), a("inv_w")
    q_lasp, pq0_lasp = a("q_laspeyres"), a("pq0")
    hh_income0, hh_spend0 = a("hh_income0"), a("hh_spend0")
    act_scale, X_exog, exog_int = a("act_scale"), a("X_exog"), a("exog_intermediate")
    own = [bool(b) for b in d["own_account"]]
    fsav0 = float(d["foreign_sav"])
    gov_dem = a("gov_dem")
    Bmat = io @ msh
    A, C, H, F = range(na), range(nc), range(nh), range(nf)
    # Complementarity, for PATH: each condition is written with the variable it
    # holds at its bound, and nothing else in the model changes.
    MCP = "least_cost" in d
    LC = {int(g["commodity"]): [int(j) for j in g["makers"]] for g in d.get("least_cost", [])}
    CAP = {int(j): float(v) for j, v in d.get("capacity", [])}
    QUOTA = {int(c): float(v) for c, v in d.get("import_quota", [])}
    CEIL = {int(c): float(b) for c, b in d.get("price_ceiling", [])}
    FLOOR = {int(c): (float(b), float(k)) for c, b, k in d.get("price_floor", [])}
    WAGE_FLOOR = d["labour"] == "wage_floor"

    # Starts at the base year, prices scaled by the numeraire, as GAMS initializes a model.
    def var(index=None, lb=None, init=0.0):
        if index is None:
            return pe.Var(bounds=(lb, None), initialize=init)
        return pe.Var(index, bounds=(lb, None), initialize=init)

    shift, er0 = (1.0, 1.0) if start == "base" else (0.85, 1.15)
    m.w = var(F, 1e-8, lambda _, f: nu if f == NUM else shift * nu)
    m.w[NUM].fix(nu)
    # Imperfect mobility: the factor has a price in each activity that uses it.
    MOB = d["mobility"]
    ismob = [False] * nf
    pairs = []
    for mb in MOB:
        f = int(mb["factor"]) - 1
        ismob[f] = True
        pairs += [(f, j) for j in A if mb["users"][j]]
    wm = {}
    if pairs:
        m.wm = var(pairs, 1e-8, shift * nu)
        wm = {k: m.wm[k] for k in pairs}

    def wf(f, j):
        return wm.get((f, j), m.w[f])

    if slack:
        m.emp = var(None, 0.0, supply[LAB])
    if MCP:
        from pyomo.mpec import ComplementarityList, complements
        m.cc = ComplementarityList()
        # Each least-cost producer's share of its commodity's output, the
        # scarcity rent per unit of a capped activity (numeraire units), the
        # premium on a quota's border price, the subsidy that holds a ceiling
        # (off the product tax rate) and the purchases that hold a floor (in
        # units of the commodity's base absorption, as solve_period counts them).
        lc_pairs = [(c, j) for c, makers in LC.items() for j in makers]
        m.share = pe.Var(lc_pairs, bounds=(0.0, None),
                         initialize=lambda _, c, j: float(msh[j, c]))
        m.rent = pe.Var(list(CAP), bounds=(0.0, None), initialize=0.0)
        m.premium = pe.Var(list(QUOTA), bounds=(0.0, None), initialize=0.0)
        m.subsidy = pe.Var(list(CEIL), bounds=(0.0, None), initialize=0.0)
        m.bought = pe.Var(list(FLOOR), bounds=(0.0, None), initialize=0.0)
        if WAGE_FLOOR:
            m.Uw = pe.Var(bounds=(0.0, None), initialize=0.0)
    rent = {j: m.rent[j] for j in CAP} if MCP else {}
    subsidy = {c: m.subsidy[c] for c in CEIL} if MCP else {}
    stock = [m.bought[c] * FLOOR[c][1] if c in FLOOR else 0.0 for c in C]

    def msh_eff(j, c):
        return m.share[c, j] if c in LC and j in LC[c] else msh[j, c]
    m.pva = var(A, 0.0, nu)
    m.pa = var(A, 1e-8, nu)
    m.pd = var(C, 1e-8, nu)
    m.arm = var(C, 1e-8, nu)
    m.dsh = var(C, 0.0, lambda _, c: domsh[c])
    m.mshm = var(C, 0.0, lambda _, c: 1.0 - domsh[c])
    m.pbase = var(C, 1e-8, nu)
    m.Aabs = var(C, 0.0, lambda _, c: A0[c])
    m.Q = var(C, 0.0, lambda _, c: Q0[c])
    m.Xraw = var(A, None, lambda _, j: X0[j])
    m.hh_income = var(H, None, lambda _, h: hh_income0[h])
    m.spend = var(H, None, lambda _, h: hh_spend0[h])
    m.savings = var(None, None, float(inv_dem.sum()) * nu)
    m.uplift_rev = var(None, None, 0.0)

    if d["foreign"] == "fixed_saving":
        m.er = var(None, 1e-8, er0)
        er = m.er
        # In dollars, foreign saving at home is the exchange rate times the target.
        fsav = er * fsav0 if d["foreign_saving_unit"] == "foreign_currency" else fsav0
    else:
        er = float(d["exchange_rate"])
        m.fsav = var(None, None, fsav0)
        fsav = m.fsav
    # World prices in foreign currency: the scenario's, off a curve.
    m.pwe = var(C, 1e-8, lambda _, c: wpe[c])
    m.pwm = var(C, 1e-8, lambda _, c: wp[c])
    for c in C:
        if not eta_x[c] > 0:
            m.pwe[c].fix(wpe[c])
        if not psi_m[c] > 0:
            m.pwm[c].fix(wp[c])
    GTARGET = d["government_target"] or ""
    GADJ = d["government_adjusts"]
    TAX_LEVER = GTARGET != "" and GADJ != "spending"
    if TAX_LEVER:
        m.tscale = var(None, 1e-6, 1.0)
        tscale = m.tscale
    else:
        tscale = 1.0

    def ta(c):
        rate = ptax_a[c] * tscale if TAX_LEVER and GADJ == "product_tax" else ptax_a[c]
        return rate - subsidy[c] if c in subsidy else rate

    def tf(c):
        rate = ptax_f[c] * tscale if TAX_LEVER and GADJ == "product_tax" else ptax_f[c]
        return rate - subsidy[c] if c in subsidy else rate

    def td(h):
        return dtax[h] * tscale if TAX_LEVER and GADJ == "direct_tax" else dtax[h]

    if d["investment"] == "fixed_real":
        m.levy = var(None, None, 0.0)           # forced saving that finances fixed investment
        levy = m.levy
    else:
        levy = 0.0

    pq = [m.pbase[c] * (1 + ta(c)) for c in C]       # price to FINAL buyers
    # Laspeyres index of purchaser prices, the numeraire at the base year.
    P = sum(pq[c] * q_lasp[c] for c in C) / float(pq0_lasp @ q_lasp)

    # Labour segments. A rigid or flexible segment employs what demand asks for, a
    # variable E set equal to it below; the informal one employs the rest of the
    # labour force, N − ΣE − U. Under a transformation each household is a pool.
    SEG = d["segments"]
    m.seg = pe.Block()
    Eseg, Useg, SUP, paid_seg = {}, [], [], {}
    for k, g in enumerate(SEG):
        hired = [int(f) - 1 for f in g["rigid"]] + [int(f) - 1 for f in g["flexible"]]
        for f in hired:
            v = pe.Var(bounds=(0.0, None), initialize=supply[f])
            m.seg.add_component(f"E{f}", v)
            Eseg[f] = v
            paid_seg[f] = v
        if g["transform"]:
            idx = [int(f) - 1 for f in g["index"]]
            dl = np.asarray(g["delta"], float)
            active = [h for h in H if g["active"][h]]
            nw = np.asarray(g["workforce"], float)
            om = float(g["omega"])
            S = {}
            for n_, h in enumerate(active):
                if g["form"] == "cet":
                    Wp = pe.Var(bounds=(1e-8, None), initialize=nu)
                    m.seg.add_component(f"Wp{k}_{n_}", Wp)
                    add(Wp ** (1 + om) == sum(dl[n_, kk] * m.w[idx[kk]] ** (1 + om)
                                              for kk in range(len(idx)) if dl[n_, kk] > 0))
                    for kk in range(len(idx)):
                        if dl[n_, kk] > 0:
                            S[(h, kk)] = dl[n_, kk] * nw[h] * (m.w[idx[kk]] / Wp) ** om
                else:
                    Zp = pe.Var(bounds=(1e-12, None), initialize=1.0)
                    m.seg.add_component(f"Zp{k}_{n_}", Zp)
                    add(Zp == sum(dl[n_, kk] * m.w[idx[kk]] ** om
                                  for kk in range(len(idx)) if dl[n_, kk] > 0))
                    for kk in range(len(idx)):
                        if dl[n_, kk] > 0:
                            S[(h, kk)] = nw[h] * dl[n_, kk] * m.w[idx[kk]] ** om / Zp
            supply_k = [sum(S[(h, kk)] for h in H if (h, kk) in S) for kk in range(len(idx))]
            for f in g["clearing"]:
                paid_seg[int(f) - 1] = supply_k[idx.index(int(f) - 1)]
            rigid = [kk for kk in range(len(idx)) if idx[kk] + 1 in g["rigid"]]
            Useg.append(sum(supply_k[kk] - Eseg[idx[kk]] for kk in rigid) if rigid else 0.0)
            SUP.append((S, supply_k))
            continue
        SUP.append(None)
        if g["harris_todaro"]:
            v = pe.Var(initialize=float(g["U0"]))
            m.seg.add_component(f"U{k}", v)
            Useg.append(v)
        else:
            Useg.append(0.0)
        paid_seg[int(g["informal"]) - 1] = (float(g["N"]) - sum(Eseg[f] for f in hired)
                                            - Useg[k])

    def paid_supply(f):
        if f in paid_seg:
            return paid_seg[f]
        if slack and f == LAB:
            return m.emp
        if WAGE_FLOOR and f == LAB:
            return supply[f] - m.Uw
        if LABOUR == "elastic_supply" and f == LAB:
            return supply[f] * (m.w[f] / P) ** float(d["labour_supply_elasticity"])
        return supply[f]

    # The output tax is inside the price of all output; the tariff inside the import price.
    pgross = [m.pd[c] * (1 + tauz[c]) / (1 + tauz0[c]) for c in C]
    if CET:
        m.pdom = var(C, 1e-8, nu)
        pdom = [m.pdom[c] for c in C]
        for c in C:
            if 0 < expsh[c] < 1:
                add(pgross[c] ** (1 + scet[c]) == (1 - expsh[c]) * pdom[c] ** (1 + scet[c])
                    + expsh[c] * (er * m.pwe[c]) ** (1 + scet[c]))
            else:
                add(pdom[c] == pgross[c])
        domfrac = [1.0 if expsh[c] <= 0 else 0.0 if expsh[c] >= 1 else
                   (1 - expsh[c]) * (pdom[c] / pgross[c]) ** scet[c] for c in C]
        expfrac = [0.0 if expsh[c] <= 0 else 1.0 if expsh[c] >= 1 else
                   expsh[c] * (er * m.pwe[c] / pgross[c]) ** scet[c] for c in C]
        pexp = [er * m.pwe[c] for c in C]
    else:
        pdom = pgross
        domfrac = [1 - expsh[c] for c in C]
        expfrac = [expsh[c] for c in C]
        pexp = pgross
    pqf = [m.pbase[c] * (1 + tf(c)) for c in C]      # price to FIRMS

    # ---- 1. value-added price
    def cesprice(pout, sh, prices, sg):
        k = [i for i in range(len(sh)) if sh[i] > 0]
        if sg == 0:
            add(pout == sum(sh[i] * prices[i] for i in k))
        elif abs(sg - 1) < 1e-9:
            add(log(pout) == sum(sh[i] * log(prices[i]) for i in k))
        else:
            add(pout ** (1 - sg) == sum(sh[i] * prices[i] ** (1 - sg) for i in k))

    def vpow(x, y, sg):
        return 1.0 if sg == 0 else (x / y) ** sg

    VINT = bool(d["vintage"])
    if VINT:
        vso, vth_o, vth_n = a("vintage_share_old"), a("vintage_theta_old"), a("vintage_theta_new")
        vs_o, vs_n, vs_t = a("vintage_sigma_old"), a("vintage_sigma_new"), a("vintage_sigma_top")
        m.pOld = var(A, 1e-8, nu)
        m.pNew = var(A, 1e-8, nu)
        for j in A:
            cesprice(m.pOld[j], vth_o[:, j], [wf(f, j) for f in F], vs_o[j])
            cesprice(m.pNew[j], vth_n[:, j], [wf(f, j) for f in F], vs_n[j])
            cesprice(m.pva[j], [vso[j], 1 - vso[j]], [m.pOld[j], m.pNew[j]], vs_t[j])
        bOld = [vso[j] * vpow(m.pva[j], m.pOld[j], vs_t[j]) for j in A]
        bNew = [(1 - vso[j]) * vpow(m.pva[j], m.pNew[j], vs_t[j]) for j in A]
    else:
        for j in A:
            if leo[j]:
                add(m.pva[j] * vunit[j] == sum(aunit[f, j] * wf(f, j) for f in F))
            elif theta[:, j].sum() <= 0:
                add(m.pva[j] == 0)
            elif abs(sva[j] - 1.0) < 1e-9:
                add(log(m.pva[j]) == sum(theta[f, j] * log(wf(f, j)) for f in F
                                         if theta[f, j] > 0))
            else:
                add(m.pva[j] ** (1 - sva[j]) == sum(theta[f, j] * wf(f, j) ** (1 - sva[j])
                                                    for f in F if theta[f, j] > 0))

    # ---- 1b. value added-energy nest
    NEST = bool(d["nest"])
    if NEST:
        emask = [bool(b) for b in d["energy_mask"]]
        vaeq, esh, thetaE = a("vae_q"), a("e_share"), a("theta_e")
        svae, sen = a("sigma_vae"), a("sigma_energy")
        iofix, carbf = a("io_fixed"), a("carb_fuel")
        p0f = a("ptax_base") + 1.0
        m.pE = var(A, 1e-8, nu)

        def fuelrel(c, j):
            return (pqf[c] + carbf[c, j]) / p0f[c]

        for j in A:
            fu = [c for c in C if thetaE[c, j] > 0]
            if not fu:
                add(m.pE[j] == nu)
            elif sen[j] == 0:
                add(m.pE[j] == sum(thetaE[c, j] * fuelrel(c, j) for c in fu))
            elif abs(sen[j] - 1) < 1e-9:
                add(log(m.pE[j]) == sum(thetaE[c, j] * log(fuelrel(c, j)) for c in fu))
            else:
                add(m.pE[j] ** (1 - sen[j]) == sum(thetaE[c, j] * fuelrel(c, j) ** (1 - sen[j])
                                                   for c in fu))

        def ces2(pout, a1, p1, b1, p2, sg):
            if sg == 0:
                add(pout == a1 * p1 + b1 * p2)
            elif abs(sg - 1) < 1e-9:
                add(log(pout) == a1 * log(p1) + b1 * log(p2))
            else:
                add(pout ** (1 - sg) == a1 * p1 ** (1 - sg) + b1 * p2 ** (1 - sg))

        if d["nest_mode"] == "value_added":
            m.pVAE = var(A, 1e-8, nu)
            for j in A:
                ces2(m.pVAE[j], 1 - esh[j], m.pva[j], esh[j], m.pE[j], svae[j])
            vaper = [vaeq[j] * (1 - esh[j]) * vpow(m.pVAE[j], m.pva[j], svae[j]) for j in A]
            eper = [vaeq[j] * esh[j] * vpow(m.pVAE[j], m.pE[j], svae[j]) for j in A]
            topcost = [vaeq[j] * m.pVAE[j] for j in A]

            def facper(f, j):
                base = ((aunit[f, j] / vunit[j] if vunit[j] != 0 else 0.0) if leo[j] else
                        (theta[f, j] * (m.pva[j] / wf(f, j)) ** sva[j] if theta[f, j] > 0
                         else 0.0))
                return base * vaper[j]

            nfree = [False] * na

            def fuelfix(c, j):
                return 0.0
        else:
            KI = int(d["capital_index"]) - 1
            nfree = [not b for b in d["nest_free"]]
            ktop, kesh, ksh = a("top_share"), a("ke_share"), a("k_share")
            sigke, sigtop = a("sigma_ke"), a("sigma_top")
            m.pKE = var(A, 1e-8, nu)
            m.pTOP = var(A, 1e-8, nu)
            for j in A:
                if nfree[j]:
                    add(m.pKE[j] == nu)
                    add(m.pTOP[j] == nu)
                    continue
                ces2(m.pKE[j], ksh[j], wf(KI, j), 1 - ksh[j], m.pE[j], sigke[j])
                fs = [f for f in F if ktop[f, j] > 0]
                if sigtop[j] == 0:
                    add(m.pTOP[j] == sum(ktop[f, j] * wf(f, j) for f in fs) + kesh[j] * m.pKE[j])
                elif abs(sigtop[j] - 1) < 1e-9:
                    add(log(m.pTOP[j]) == sum(ktop[f, j] * log(wf(f, j)) for f in fs)
                        + kesh[j] * log(m.pKE[j]))
                else:
                    add(m.pTOP[j] ** (1 - sigtop[j]) == sum(ktop[f, j] * wf(f, j) ** (1 - sigtop[j])
                                                            for f in fs)
                        + kesh[j] * m.pKE[j] ** (1 - sigtop[j]))
            keper = [vaeq[j] * kesh[j] * vpow(m.pTOP[j], m.pKE[j], sigtop[j]) for j in A]
            eper = [0.0 if nfree[j] else keper[j] * (1 - ksh[j]) * vpow(m.pKE[j], m.pE[j], sigke[j])
                    for j in A]
            topcost = [sum(aunit[f, j] * wf(f, j) for f in F)
                       + sum((pqf[c] + carbf[c, j]) * io[c, j] for c in C if emask[c])
                       if nfree[j] else vaeq[j] * m.pTOP[j] for j in A]

            def facper(f, j):
                if nfree[j]:
                    return aunit[f, j]
                if f == KI:
                    return keper[j] * ksh[j] * vpow(m.pKE[j], wf(f, j), sigke[j])
                return vaeq[j] * ktop[f, j] * vpow(m.pTOP[j], wf(f, j), sigtop[j])

            def fuelfix(c, j):
                return io[c, j] if nfree[j] and emask[c] else 0.0

        def iouse(c, j):
            if not emask[c]:
                return iofix[c, j]
            if nfree[j]:
                return fuelfix(c, j)
            if thetaE[c, j] <= 0:
                return 0.0
            if sen[j] == 0:
                return eper[j] * thetaE[c, j] / p0f[c]
            return eper[j] * thetaE[c, j] * (m.pE[j] / fuelrel(c, j)) ** sen[j] / p0f[c]

    # ---- 2-3. unit cost of each activity and domestic price of each commodity
    for j in A:
        cost = (sum(iofix[c, j] * pqf[c] for c in C) + topcost[j] if NEST
                else sum(io[c, j] * pqf[c] for c in C if io[c, j] != 0) + m.pva[j] * vunit[j])
        add(m.pa[j] == cost + atax[j] + (rent[j] if j in rent else 0.0))
    for c in C:
        if c in LC:
            # Its price is its own variable, set by the producers in use:
            # unit cost − price >= 0 ⊥ share >= 0, and the shares make the output.
            for j in LC[c]:
                m.cc.add(complements((m.pa[j] - m.pd[c]) / nu >= 0, m.share[c, j] >= 0))
            add(sum(m.share[c, j] for j in LC[c]) == 1.0)
            continue
        add(m.pd[c] == sum(msh[j, c] * m.pa[j] for j in A if msh[j, c] != 0))

    # ---- 4-5. Armington
    pm = [er * m.pwm[c] * (1 + tm[c] + (m.premium[c] if c in QUOTA else 0.0)) / (1 + tm0[c])
          for c in C]
    for c in C:
        if abs(sarm[c] - 1) < 1e-9:
            add(log(m.arm[c]) == domsh[c] * log(pdom[c]) + (1 - domsh[c]) * log(pm[c]))
        else:
            add(m.arm[c] ** (1 - sarm[c]) == domsh[c] * pdom[c] ** (1 - sarm[c])
                + (1 - domsh[c]) * pm[c] ** (1 - sarm[c]))
        add(m.dsh[c] == domsh[c] * (m.arm[c] / pdom[c]) ** sarm[c])
        add(m.mshm[c] == (1 - domsh[c]) * (m.arm[c] / pm[c]) ** sarm[c])

    # ---- 6. delivered price: margins are themselves priced at delivered prices
    for c in C:
        add(m.pbase[c] == m.arm[c] * armsh[c] + sum(pqf[k] * mq[k, c] for k in C if mq[k, c] != 0))

    # ---- 6b. pools of an imperfectly mobile factor
    offer, mob_income = {}, {}
    m.mob = pe.Block()
    for n_, mb in enumerate(MOB):
        f = int(mb["factor"]) - 1
        om = float(mb["omega"])
        dl = np.asarray(mb["delta"], float)
        L = np.asarray(mb["endowment"], float)
        users = mb["users"]
        npool = len(L)

        def reach(o, dl=dl, users=users):
            return [j for j in A if users[j] and dl[o, j] > 0]

        if mb["form"] == "cet":
            Wp = pe.Var(range(npool), bounds=(1e-8, None), initialize=nu)
            m.mob.add_component(f"Wp{n_}", Wp)
            for o in range(npool):
                add(Wp[o] ** (1 + om) == sum(dl[o, j] * wf(f, j) ** (1 + om) for j in reach(o)))

            def S(o, j, f=f, dl=dl, L=L, Wp=Wp, om=om):
                return dl[o, j] * L[o] * (wf(f, j) / Wp[o]) ** om

            mob_income[f] = sum(Wp[o] * L[o] for o in range(npool))
        else:
            Z = pe.Var(range(npool), bounds=(1e-12, None), initialize=1.0)
            m.mob.add_component(f"Z{n_}", Z)
            for o in range(npool):
                add(Z[o] == sum(dl[o, j] * wf(f, j) ** om for j in reach(o)))

            def S(o, j, f=f, dl=dl, L=L, Z=Z, om=om):
                return L[o] * dl[o, j] * wf(f, j) ** om / Z[o]

            mob_income[f] = sum(wf(f, j) * S(o, j) for o in range(npool) for j in reach(o))
        for j in A:
            if users[j]:
                offer[(f, j)] = sum(S(o, j) for o in range(npool) if dl[o, j] > 0)
        add(m.w[f] * float(L.sum()) == mob_income[f])

    # Rents times the limit and the quota: at the solution they are rent times
    # output and premium times imports, as solve_period books them.
    quota_rent = sum(m.premium[c] * er * m.pwm[c] * q for c, q in QUOTA.items()) if MCP else 0.0
    extra_income = [0.0] * nf
    if CAP:
        extra_income[int(d["capacity_owner"])] += sum(m.rent[j] * lim for j, lim in CAP.items())
    if QUOTA and d["quota_rent_to"] is not None:
        extra_income[int(d["quota_rent_to"])] += quota_rent

    def factor_income(f):
        return (mob_income[f] if ismob[f] else m.w[f] * paid_supply(f)) + extra_income[f]

    # ---- 7. households, with the wedge recycled
    direct = [sum(hh_direct[h, j] * m.pa[j] * act_scale[j] for j in A if hh_direct[h, j] != 0)
              for h in H]

    def segment_income(h):
        tot = 0.0
        for k, g in enumerate(SEG):
            idx = [int(f) - 1 for f in g["index"]]
            sh = np.asarray(g["shares"], float)
            workforce = np.asarray(g["workforce"], float)
            queue = np.asarray(g["queue"], float)
            rigid = [int(f) - 1 for f in g["rigid"]]
            if g["transform"]:
                S, supply_k = SUP[k]
                for kk, f in enumerate(idx):
                    if (h, kk) not in S:
                        continue
                    part = S[(h, kk)] * Eseg[f] / supply_k[kk] if f in rigid else S[(h, kk)]
                    tot += m.w[f] * part * (1 - gsh[f] - rsh[f])
                continue
            hired = rigid + [int(f) - 1 for f in g["flexible"]]
            employed = 0.0
            for kk, f in enumerate(idx):
                if f not in hired:
                    continue
                tot += m.w[f] * sh[h, kk] * Eseg[f] * (1 - gsh[f] - rsh[f])
                employed += sh[h, kk] * Eseg[f]
            b = int(g["informal"]) - 1
            tot += m.w[b] * (workforce[h] - employed - queue[h] * Useg[k]) * (1 - gsh[b] - rsh[b])
        return tot

    for h in H:
        add(m.hh_income[h] == sum(facshare[h, f] * factor_income(f) * (1 - gsh[f] - rsh[f])
                                  for f in F if facshare[h, f] != 0)
            + segment_income(h) + transfers[h] - levy * popsh[h] + m.uplift_rev * alpha[h]
            + gov_tr[h] + remit[h])
        # Spending comes out of DISPOSABLE income: direct tax first, saving after.
        add(m.spend[h] == m.hh_income[h] * (1 - td(h)) * (1 - savrate[h]) - direct[h])

    scarb = a("carbon_household")
    if np.any(scarb != 0):
        # A specific carbon tax on households: the aggregated form does not hold.
        shr = beta_les if les else hh_budget
        effp = [[pq[c] * (1 + Wt[h, c]) + scarb[c] for c in C] for h in H]
        if les:
            sup = [m.spend[h] - sum(effp[h][c] * gamma[h, c] for c in C) for h in H]
        else:
            sup = [m.spend[h] for h in H]
        hhq = [[gamma[h, c] + shr[h, c] * sup[h] / effp[h][c] for c in C] for h in H]
        add(m.uplift_rev == sum(hhq[h][c] * W[h, c] * pq[c] for h in H for c in C))
        gov_cost = -sum(hhq[h][c] * (Wg[h, c] * pq[c] + scarb[c]) for h in H for c in C) + gov_absorbs
        hh_final = [sum(hhq[h][c] for h in H) for c in C]
    else:
        if les:
            sup = [m.spend[h] - sum(pq[c] * (1 + Wt[h, c]) * gamma[h, c] for c in C) for h in H]
        else:
            sup = [m.spend[h] for h in H]
        hh_val = [sum(coef[h, c] * sup[h] for h in H) for c in C]
        # The recycling series: credited revenue raises spending, which raises the wedge.
        add(m.uplift_rev == sum(g_cap[h] * sup[h] for h in H) + sum(gam_W[c] * pq[c] for c in C))
        gov_cost = -(sum(g_gov[h] * sup[h] for h in H) + sum(gam_Wg[c] * pq[c] for c in C)) + gov_absorbs
        hh_final = [gamma_sum[c] + hh_val[c] / pq[c] for c in C]

    if GTARGET == "saving" and GADJ == "spending":
        gov_w = a("gov_w")
        m.gbud = var(None, None, float(gov_dem.sum()) * nu)
        gov_q = [gov_w[c] * m.gbud / pq[c] for c in C]
    elif d["government"] == "real_fixed":
        gov_q = list(a("gov_q"))
    elif d["government"] == "saving_rate":
        gov_w = a("gov_w")
        m.gbud = var(None, None, float(gov_dem.sum()) * nu)
        gov_q = [gov_w[c] * m.gbud / pq[c] for c in C]
    else:
        gov_q = [gov_dem[c] * nu / pq[c] for c in C]
    gov_spend = sum(pq[c] * gov_q[c] for c in C)
    hh_sav = sum(m.hh_income[h] * (1 - td(h)) * savrate[h] for h in H)
    gov_inst = (sum(m.hh_income[h] * td(h) for h in H)
                + sum(factor_income(f) * gsh[f] for f in F if gsh[f] != 0) - float(gov_tr.sum()))
    if QUOTA and d["quota_rent_to"] is None:
        gov_inst = gov_inst + quota_rent                 # auctioned licences
    support = sum(pq[c] * stock[c] for c in FLOOR)

    if d["investment"] == "savings_driven":
        inv_q = [inv_w[c] * m.savings / pqf[c] for c in C]
    else:
        inv_q = list(inv_q_fix)
        add(m.savings == sum(pqf[c] * inv_q[c] for c in C))

    # ---- 8. absorption balance, in place of the Leontief inversion
    X = [X_exog[j] if own[j] else m.Xraw[j] for j in A]
    for c in C:
        if NEST:
            inter = sum(iouse(c, j) * (m.Xraw[j] + X_exog[j]) for j in A) - exog_int[c]
        elif LC:
            inter = sum(io[c, j] * m.Xraw[j] for j in A if io[c, j] != 0)
        else:
            inter = sum(Bmat[c, k] * m.Q[k] for k in C if Bmat[c, k] != 0)
        add(m.Aabs[c] == inter + sum(mq[c, k] * m.Aabs[k] for k in C if mq[c, k] != 0)
            + hh_final[c] + gov_q[c] + inv_q[c] + exog_int[c] + stock[c])
        if expsh[c] >= 1:
            add(m.Q[c] == 0)
        else:
            add(m.Q[c] == m.dsh[c] * armsh[c] / (domfrac[c] * (1 + tauz0[c])) * m.Aabs[c])
    for j in A:
        add(m.Xraw[j] == sum(msh_eff(j, c) * m.Q[c] for c in C
                             if msh[j, c] != 0 or (c in LC and j in LC[c])))

    # ---- 9. saving
    fin_c = [hh_final[c] + gov_q[c] + stock[c] for c in C]
    tax_rev = (sum(m.pbase[c] * ((m.Aabs[c] - fin_c[c]) * tf(c) + fin_c[c] * ta(c)) for c in C)
               + sum(atax[j] * X[j] for j in A if atax[j] != 0)
               + (sum(carbf[c, j] * iouse(c, j) * X[j] for c in C for j in A if carbf[c, j] != 0)
                  if NEST else 0.0)
               + sum(tm[c] * er * m.pwm[c] * m.mshm[c] * armsh[c] * m.Aabs[c] / (1 + tm0[c])
                     for c in C if tm[c] != 0)
               + sum(tauz[c] * m.pd[c] * m.Q[c] for c in C if tauz[c] != 0))
    # Purchases that hold a price floor come out of government saving.
    add(m.savings == hh_sav + levy + tax_rev + gov_inst - gov_spend - gov_cost - support + fsav)
    if d["government"] == "saving_rate":
        add(m.gbud == (1 - float(d["gov_savrate"])) * (tax_rev + gov_inst - gov_cost))
    GTVAL = float(d["government_target_value"])
    # Revenue counts the product tax before the subsidy that holds a ceiling.
    subsidy_cost = sum(m.subsidy[c] * m.pbase[c] * m.Aabs[c] for c in CEIL) if MCP else 0.0
    if GTARGET == "saving" and GADJ == "spending":
        add(m.gbud == tax_rev + gov_inst - gov_cost - support - GTVAL)
    elif TAX_LEVER and GTARGET == "saving":
        add(tax_rev + gov_inst - gov_spend - gov_cost - support == GTVAL)
    elif TAX_LEVER and GTARGET == "revenue":
        add(tax_rev + subsidy_cost + gov_inst + float(gov_tr.sum()) - gov_cost + gov_absorbs
            == GTVAL)

    # ---- 10. factor demand
    def facint(f, j):
        if VINT:
            out = 0.0
            if vth_o[f, j] > 0:
                out += vth_o[f, j] * vpow(m.pOld[j], wf(f, j), vs_o[j]) * bOld[j]
            if vth_n[f, j] > 0:
                out += vth_n[f, j] * vpow(m.pNew[j], wf(f, j), vs_n[j]) * bNew[j]
            return out
        if leo[j]:
            return aunit[f, j] / vunit[j] if vunit[j] != 0 else 0.0
        return theta[f, j] * (m.pva[j] / wf(f, j)) ** sva[j] if theta[f, j] > 0 else 0.0

    def facdem_j(f, j):
        if NEST:
            return facper(f, j) * X[j]
        unit = facint(f, j)
        return 0.0 if isinstance(unit, float) and unit == 0.0 else unit * vunit[j] * X[j]

    facdem = [sum(facdem_j(f, j) for j in A) for f in F]
    if slack:
        add(m.emp == facdem[LAB])

    # ---- 11. the markets solved for. The numeraire's is left to Walras' law.
    for f in F:
        if f == NUM:
            continue
        if ismob[f]:
            for j in A:
                if (f, j) in offer:
                    add(facdem_j(f, j) == offer[(f, j)])
            continue
        if f in paid_seg:
            add(facdem[f] == paid_seg[f])
            continue
        if LABOUR == "fixed_real_wage" and f == LAB:
            continue
        if WAGE_FLOOR and f == LAB:
            # Employment is demand; the floor: real wage − floor >= 0 ⊥ U >= 0.
            add(facdem[f] == supply[f] - m.Uw)
            floor_w = float(d["real_wage_target"])
            m.cc.add(complements(m.w[f] / (floor_w * P) - 1.0 >= 0, m.Uw >= 0))
            continue
        if LABOUR == "wage_curve" and f == LAB:
            u0 = float(d["natural_unemployment"])
            add(log(m.w[f] / P) + float(d["wage_curve_elasticity"])
                * log((1 - facdem[f] / (supply[f] / (1 - u0))) / u0) == 0)
        elif LABOUR == "elastic_supply" and f == LAB:
            add(facdem[f] == supply[f] * (m.w[f] / P) ** float(d["labour_supply_elasticity"]))
        else:
            add(facdem[f] == supply[f])
    if LABOUR == "fixed_real_wage":
        add(m.w[LAB] == float(d["real_wage_target"]) * P)
    for k, g in enumerate(SEG):
        b = int(g["informal"]) - 1
        for f, target in zip(g["rigid"], g["target"]):
            add(m.w[int(f) - 1] == float(target) * P)
        for f in g["flexible"]:
            add(m.w[int(f) - 1] == m.w[b])
        if g["harris_todaro"]:
            rigid = [int(f) - 1 for f in g["rigid"]]
            add(m.w[b] * (sum(Eseg[f] for f in rigid) + Useg[k])
                == float(g["lam"]) * sum(m.w[f] * Eseg[f] for f in rigid))

    imports = [m.mshm[c] * armsh[c] * m.Aabs[c] / (1 + tm0[c]) for c in C]
    exports = [expfrac[c] * (1 + tauz0[c]) * m.Q[c] for c in C]
    trade = sum(er * m.pwm[c] * imports[c] for c in C) - sum(pexp[c] * exports[c] for c in C)
    # ---- 12. the curves
    for c in C:
        if eta_x[c] > 0:
            add(exports[c] == E0[c] * (m.pwe[c] / wpe[c]) ** (-eta_x[c]))
        if psi_m[c] > 0:
            add(imports[c] == M0[c] * (m.pwm[c] / wp[c]) ** psi_m[c])
    current = trade + sum(factor_income(f) * rsh[f] for f in F if rsh[f] != 0) - float(remit.sum())
    add(current == fsav)

    # ---- 13. complementarity, each condition scaled as solve_period scales it
    for j, lim in CAP.items():
        m.cc.add(complements((lim - X[j]) / max(lim, 1.0) >= 0, m.rent[j] >= 0))
    for c, q in QUOTA.items():
        m.cc.add(complements((q - imports[c]) / max(q, 1.0) >= 0, m.premium[c] >= 0))
    for c, b in CEIL.items():
        bound = b * P * (1.0 + float(d["ptax_base"][c]))
        m.cc.add(complements((bound - pq[c]) / bound >= 0, m.subsidy[c] >= 0))
    for c, (b, _) in FLOOR.items():
        bound = b * P * (1.0 + float(d["ptax_base"][c]))
        m.cc.add(complements((pq[c] - bound) / bound >= 0, m.bought[c] >= 0))

    if not MCP:
        m.obj = pe.Objective(expr=0.0)
    handles = dict(w=m.w, wm=wm, er=er, fsav=fsav, pwe=m.pwe, pwm=m.pwm, tscale=tscale,
                   levy=levy, U=Useg)
    if MCP:
        handles.update(pd=m.pd, share=m.share, rent=m.rent, premium=m.premium,
                       subsidy=m.subsidy, bought=m.bought)
    return m, handles
