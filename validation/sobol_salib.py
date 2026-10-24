"""sobol_salib.py — cgelab's Sobol indices against SALib, on the same design.

SALib (Herman and Usher, 2017; Iwanaga, Usher and Herman, 2022) is the reference implementation
of Saltelli's design and of the estimators cgelab uses: first order by Saltelli et al. (2010),
total by Jansen (1999). Two checks:

1. the design: SALib's ``sample.sobol`` with the same seed draws the same points as
   ``cgelab.sobol``;
2. the estimators: SALib's ``analyze.sobol``, given cgelab's outputs in SALib's row order, gives
   the same first-order and total indices.

Both on the Ishigami function, whose indices are known in closed form, and on the toy economy's
real GDP response to two elasticities.

    pip install SALib
    python validation/sobol_salib.py
"""
from __future__ import annotations

import sys

import numpy as np

import cgelab as cge
from cgelab.examples import toy

try:
    from SALib.analyze import sobol as salib_analyze
    from SALib.sample import sobol as salib_sample
except ImportError:                                   # pragma: no cover
    sys.exit("SALib is not installed: pip install SALib")


def salib_order(result, output):
    """cgelab's outputs in SALib's row order: for each base point, A, every AB_i, then B."""
    A, B, AB = result._arrays(output)
    return np.column_stack([A, AB, B]).ravel()


def compare(name, run, parameters, n, bounds):
    res = cge.sobol(run, parameters, n=n, bootstrap=0)
    problem = {"num_vars": len(parameters), "names": list(parameters), "bounds": bounds}
    sample = salib_sample.sample(problem, n, calc_second_order=False, scramble=True,
                                 seed=res.seed)
    X = res.points
    ours = np.column_stack([X.loc[X["matrix"] == "A", list(parameters)].to_numpy(),
                            *[X.loc[X["matrix"] == f"AB_{k}", list(parameters)].to_numpy()
                              for k in parameters],
                            X.loc[X["matrix"] == "B", list(parameters)].to_numpy()])
    ours = ours.reshape(-1, len(parameters))
    design = float(np.max(np.abs(ours - sample)))
    worst = 0.0
    idx = res.indices()
    for out in res.outputs:
        Si = salib_analyze.analyze(problem, salib_order(res, out), calc_second_order=False,
                                   num_resamples=2, print_to_console=False, seed=1)
        mine = idx.loc[out]
        worst = max(worst, float(np.max(np.abs(mine["S1"].to_numpy() - Si["S1"]))),
                    float(np.max(np.abs(mine["ST"].to_numpy() - Si["ST"]))))
    print(f"  {name:<22} design {design:.1e}   indices {worst:.1e}")
    print(idx[["S1", "ST"]].round(4).to_string())
    return design, worst


def main() -> int:
    pi = np.pi
    ishigami = {f"x{i}": cge.Uniform(-pi, pi) for i in (1, 2, 3)}

    def f(v):
        return (np.sin(v["x1"]) + 7.0 * np.sin(v["x2"]) ** 2
                + 0.1 * v["x3"] ** 4 * np.sin(v["x1"]))

    print("cgelab.sobol against SALib")
    results = [compare("Ishigami", f, ishigami, 4096, [[-pi, pi]] * 3)]

    sam, econ = toy.build()
    run = cge.spec_sweep(sam, toy.spec(), scenario=cge.Scenario(product_tax=econ.ptax * 1.5),
                         measure=lambda b, a: {"gdp_pct": 100 * (
                             cge.real_gdp(a)["expenditure"] / cge.real_gdp(b)["expenditure"] - 1)})
    params = {"sigma_va": cge.Uniform(0.3, 1.3), "sigma_armington": cge.Uniform(1.5, 3.5)}
    results.append(compare("toy economy, real GDP", run, params, 256, [[0.3, 1.3], [1.5, 3.5]]))

    worst = max(max(r) for r in results)
    ok = worst < 1e-12
    print(f"{'OK' if ok else 'FAIL'}: worst difference {worst:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
