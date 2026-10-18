"""Benchmark of Hosoe, Gasawa and Hashimoto (2004), chapter 5.

The social accounting matrix is the one printed in the GAMS model library file
``splcge``. The published program solves that benchmark and nothing else, so the
check is the benchmark: output 15 and 35, prices equal to one, Walras at zero.
"""

import pandas as pd
import pytest

import cgelab as cge


def hosoe_economy():
    entries = [
        ("c_brd", "hh_1", 15.0), ("c_mlk", "hh_1", 35.0),
        ("f_cap", "a_brd", 5.0), ("f_cap", "a_mlk", 20.0),
        ("f_lab", "a_brd", 10.0), ("f_lab", "a_mlk", 15.0),
        ("hh_1", "f_cap", 25.0), ("hh_1", "f_lab", 25.0),
        ("a_brd", "c_brd", 15.0), ("a_mlk", "c_mlk", 35.0),
        # The published SAM has no government, investment or rest of the world.
        # The package requires the three accounts, so they are entered empty.
        ("gov", "gov", 0.0), ("s_i", "s_i", 0.0), ("row", "row", 0.0),
    ]
    sam = cge.SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]),
                            name="hosoe")
    sam.check_balance(abs_tol=1e-9)
    spec = cge.ModelSpec(
        activities=["a_brd", "a_mlk"], commodities=["c_brd", "c_mlk"],
        households=["hh_1"], factors=("f_lab", "f_cap"),
        government="gov", investment="s_i", rest_of_world="row",
        sigma_va=1.0, margins=False, numeraire_factor="f_lab")
    return cge.calibrate(sam, spec)


def test_the_published_benchmark_reproduces():
    econ = hosoe_economy()
    base = cge.solve_period(econ)
    assert base.converged
    assert abs(base.walras) < 1e-9
    assert base.X[econ.activity("a_brd")] == pytest.approx(15.0, abs=1e-8)
    assert base.X[econ.activity("a_mlk")] == pytest.approx(35.0, abs=1e-8)
    assert base.factor_prices[econ.factor("f_lab")] == pytest.approx(1.0, abs=1e-8)
    assert base.pq[econ.commodity("c_brd")] == pytest.approx(1.0, abs=1e-8)
    assert base.imports.sum() == pytest.approx(0.0, abs=1e-8)
    assert base.savings == pytest.approx(0.0, abs=1e-8)
