"""Time, memory and convergence of ``solve_period`` on SAMs of increasing size.

The SAMs are synthetic and balanced by construction, with a fixed seed, so a run
is reproducible and needs no data: ``national(n)`` has ``n`` sectors, two
factors, two households, a government, investment and the rest of the world;
``regional(r, c)`` has ``r`` regions of ``c`` goods each, traded between regions
under the origin nest. Each size is calibrated, solved at the base year and
solved again after a shock to productivity and world prices.

For each solve the script records wall time, the number of residual
evaluations inside the root finder (zero when the start point already solves
the system, as the base year does), time per evaluation, the start that converged, the residual, the
distance from the base year (which must be zero at the base) and the peak of
Python memory allocations. ``--profile`` adds the functions where the time of
the largest shocked solve goes.

    python tools/benchmark_solver.py                      # both ladders
    python tools/benchmark_solver.py --national 10 40 160 --regional 2x4 4x8
    python tools/benchmark_solver.py --out bench.csv --profile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager

import numpy as np
import pandas as pd

import cgelab as cge
import cgelab.regional as _regional
import cgelab.solve as _solve

SEED = 20261030


# ---------------------------------------------------------------------------
# Synthetic SAMs
# ---------------------------------------------------------------------------

def _intermediates(rng, n, output, share=0.4):
    """Input coefficients whose row use stays below output, so final demand is positive."""
    a = rng.uniform(0.0, 1.0, (n, n)) * (rng.uniform(size=(n, n)) < min(1.0, 6.0 / n))
    np.fill_diagonal(a, a.diagonal() + 0.2)
    use = a * output[None, :]
    # Each row uses at most `share` of that commodity's output.
    use *= share * output[:, None] / np.maximum(use.sum(axis=1, keepdims=True), 1e-12)
    return use


def _ras(pattern, rows, cols, tol=1e-12, max_iter=10_000):
    m = pattern.copy()
    for _ in range(max_iter):
        m *= (rows / m.sum(axis=1))[:, None]
        m *= (cols / m.sum(axis=0))[None, :]
        if np.max(np.abs(m.sum(axis=1) - rows)) < tol * rows.max():
            return m
    raise RuntimeError("RAS did not converge on the final-demand split")


def national(n: int, seed: int = SEED):
    """A balanced ``n``-sector SAM and its spec."""
    rng = np.random.default_rng(seed + n)
    acts = [f"a{i}" for i in range(n)]
    coms = [f"c{i}" for i in range(n)]
    x = rng.uniform(50.0, 150.0, n)
    z = _intermediates(rng, n, x)                         # z[c, a]
    va = x - z.sum(axis=0)
    trade = 0.1 * x                                      # exports equal imports, by commodity
    final = x - z.sum(axis=1)                            # what is left for final demand
    cap_share = rng.uniform(0.3, 0.6, n)
    gdp = va.sum()
    gov = 0.1 * gdp                                      # paid out of capital income
    income = gdp - gov
    cons = 0.75 * income
    inv = income - cons
    split = _ras(rng.uniform(0.5, 1.5, (n, 4)), final,
                 np.array([0.6 * cons, 0.4 * cons, gov, inv]))
    rows = []
    for j in range(n):
        rows += [("f_lab", acts[j], va[j] * (1 - cap_share[j])),
                 ("f_cap", acts[j], va[j] * cap_share[j]),
                 (acts[j], coms[j], x[j]),
                 ("row", coms[j], trade[j]), (coms[j], "row", trade[j])]
        rows += [(coms[i], acts[j], z[i, j]) for i in range(n) if z[i, j] > 0]
        rows += [(coms[j], "hh1", split[j, 0]), (coms[j], "hh2", split[j, 1]),
                 (coms[j], "gov", split[j, 2]), (coms[j], "s_i", split[j, 3])]
    lab, cap = (va * (1 - cap_share)).sum(), (va * cap_share).sum()
    rows += [("hh1", "f_lab", 0.4 * lab), ("hh2", "f_lab", 0.6 * lab),
             ("hh1", "f_cap", cap - gov), ("gov", "f_cap", gov),
             ("s_i", "hh1", (0.4 * lab + cap - gov) - 0.6 * cons),
             ("s_i", "hh2", 0.6 * lab - 0.4 * cons)]
    sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]),
                            name=f"national-{n}")
    spec = cge.ModelSpec(activities=acts, commodities=coms, households=["hh1", "hh2"],
                         factors=("f_lab", "f_cap"), numeraire_factor="f_lab",
                         government="gov", investment="s_i", rest_of_world="row",
                         sigma_va=0.8, sigma_armington=2.0, margins=False)
    return sam, spec


def regional(r: int, c: int, seed: int = SEED):
    """A balanced SAM of ``r`` regions and ``c`` goods each, with interregional trade."""
    rng = np.random.default_rng(seed + 1000 * r + c)
    regs = [f"r{k}" for k in range(r)]
    acts = [f"{g}:a{i}" for g in regs for i in range(c)]
    coms = [f"{g}:c{i}" for g in regs for i in range(c)]
    n = r * c
    x = rng.uniform(50.0, 150.0, n)
    # Home bias: two thirds of each local good's purchases stay in the region.
    home = np.full((r, r), 1.0 / (3.0 * max(r - 1, 1)))
    np.fill_diagonal(home, 2.0 / 3.0 if r > 1 else 1.0)
    z = np.zeros((n, n))                                 # purchases of origin commodity by activity
    for d in range(r):
        cols = slice(d * c, (d + 1) * c)
        local = _intermediates(rng, c, x[cols], share=0.35)   # local good by activity in d
        for o in range(r):
            z[o * c:(o + 1) * c, cols] = home[o, d] * local
    va = x - z.sum(axis=0)
    trade = 0.1 * x
    final = x - z.sum(axis=1)
    cap_share = rng.uniform(0.3, 0.6, n)
    gdp_r = np.array([va[k * c:(k + 1) * c].sum() for k in range(r)])
    gov = 0.1 * gdp_r.sum()
    cap_r = np.array([(va * cap_share)[k * c:(k + 1) * c].sum() for k in range(r)])
    gov_r = gov * cap_r / cap_r.sum()
    income_r = gdp_r - gov_r
    cons_r = 0.75 * income_r
    inv = (income_r - cons_r).sum()
    pattern = rng.uniform(0.5, 1.5, (n, r + 2))
    for o in range(r):
        pattern[o * c:(o + 1) * c, :r] *= home[o][None, :] * r
    split = _ras(pattern, final, np.concatenate([cons_r, [gov, inv]]))
    rows = []
    for j in range(n):
        g = regs[j // c]
        rows += [(f"{g}:f_lab", acts[j], va[j] * (1 - cap_share[j])),
                 (f"{g}:f_cap", acts[j], va[j] * cap_share[j]),
                 (acts[j], coms[j], x[j]),
                 ("row", coms[j], trade[j]), (coms[j], "row", trade[j])]
        rows += [(coms[i], acts[j], z[i, j]) for i in range(n) if z[i, j] > 0]
        rows += [(coms[j], f"{regs[d]}:hh", split[j, d]) for d in range(r)]
        rows += [(coms[j], "gov", split[j, r]), (coms[j], "s_i", split[j, r + 1])]
    for k, g in enumerate(regs):
        part = slice(k * c, (k + 1) * c)
        lab = (va * (1 - cap_share))[part].sum()
        rows += [(f"{g}:hh", f"{g}:f_lab", lab), (f"{g}:hh", f"{g}:f_cap", cap_r[k] - gov_r[k]),
                 ("gov", f"{g}:f_cap", gov_r[k]), ("s_i", f"{g}:hh", income_r[k] - cons_r[k])]
    sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]),
                            name=f"regional-{r}x{c}")
    spec = cge.ModelSpec(regions=tuple(regs), activities=acts, commodities=coms,
                         households=[f"{g}:hh" for g in regs],
                         factors=tuple(f"{g}:{f}" for g in regs for f in ("f_lab", "f_cap")),
                         numeraire_factor=f"{regs[0]}:f_lab", government="gov",
                         investment="s_i", rest_of_world="row", sigma_va=0.8,
                         sigma_armington=2.0, sigma_interregional=2.0, margins=False)
    return sam, spec


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@contextmanager
def _counting():
    """Count residual evaluations by wrapping the root finder both solvers call."""
    calls = {"n": 0}
    originals = (_solve.root, _regional.root)

    def counted(fun, x0, *args, **kwargs):
        def wrapped(z):
            calls["n"] += 1
            return fun(z)
        return originals[0](wrapped, x0, *args, **kwargs)

    _solve.root = _regional.root = counted
    try:
        yield calls
    finally:
        _solve.root, _regional.root = originals


def _shock(econ, rng):
    n_local = (econ.regional.n_local if econ.regional is not None else econ.n_com)
    return cge.Scenario(productivity=rng.uniform(0.95, 1.05, econ.n_act),
                        world_price=rng.uniform(0.9, 1.1, n_local))


def measure(label, sam, spec, accounts, profile=None):
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    econ = cge.calibrate(sam, spec)
    t_cal = time.perf_counter() - t0
    out = []
    for kind, scenario in (("base", cge.Scenario()), ("shock", _shock(econ, rng))):
        with _counting() as calls:
            t0 = time.perf_counter()
            res = cge.solve_period(econ, scenario=scenario, raise_on_fail=False)
            wall = time.perf_counter() - t0
        # Memory in a second solve: tracing every allocation slows the first
        # one down by a factor of three, which would be measured as solver time.
        tracemalloc.start()
        cge.solve_period(econ, scenario=scenario, raise_on_fail=False)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        if profile is not None and kind == "shock":
            prof = cProfile.Profile()
            prof.enable()
            cge.solve_period(econ, scenario=scenario, raise_on_fail=False)
            prof.disable()
            profile[label] = prof
        out.append(dict(
            model=label, accounts=accounts, unknowns=int(np.size(res.z)), solve=kind,
            calibrate_s=t_cal, wall_s=wall, evaluations=calls["n"],
            ms_per_eval=1e3 * wall / calls["n"] if calls["n"] else np.nan, start=res.attempt,
            converged=bool(res.converged), residual=float(res.residual),
            walras=float(res.walras),
            base_gap=float(np.max(np.abs(res.X / econ.X0 - 1.0))) if kind == "base" else np.nan,
            peak_mb=peak / 2**20))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--national", type=int, nargs="*", default=[5, 10, 20, 40, 80, 160])
    parser.add_argument("--regional", nargs="*", default=["2x3", "2x6", "4x6", "4x12", "8x12"])
    parser.add_argument("--out", help="write the table to this CSV file")
    parser.add_argument("--profile", action="store_true",
                        help="profile the largest shocked solve of each ladder")
    args = parser.parse_args(argv)

    rows, profiles = [], {}
    for n in args.national:
        sam, spec = national(n)
        accounts = len(sam.accounts)
        print(f"national {n} ...", file=sys.stderr, flush=True)
        rows += measure(f"national-{n}", sam, spec, accounts,
                        profiles if args.profile and n == max(args.national) else None)
    sizes = [tuple(map(int, s.split("x"))) for s in args.regional]
    for r, c in sizes:
        sam, spec = regional(r, c)
        print(f"regional {r}x{c} ...", file=sys.stderr, flush=True)
        largest = max(sizes, key=lambda s: s[0] * s[1])
        rows += measure(f"regional-{r}x{c}", sam, spec, len(sam.accounts),
                        profiles if args.profile and (r, c) == largest else None)

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3g}".format):
        print(table.drop(columns=["walras"]).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
    for label, prof in profiles.items():
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(18)
        print(f"\n--- where the time goes: {label}, shocked solve ---")
        print("\n".join(buf.getvalue().splitlines()[:45]))
    failed = table[~table.converged | (table.base_gap > 1e-8)]
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
