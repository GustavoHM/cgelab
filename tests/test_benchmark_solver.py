"""The benchmark's synthetic SAMs balance, calibrate and reproduce their base year."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import cgelab as cge

TOOL = Path(__file__).resolve().parents[1] / "tools" / "benchmark_solver.py"
spec = importlib.util.spec_from_file_location("benchmark_solver", TOOL)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("build", [lambda: bench.national(8), lambda: bench.regional(3, 4)])
def test_synthetic_sams_balance_and_solve_at_the_base_year(build):
    sam, model_spec = build()
    assert sam.check_balance(abs_tol=1e-8)["ok"]
    econ = cge.calibrate(sam, model_spec)
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-8
    np.testing.assert_allclose(result.X, econ.X0, rtol=1e-9)
    np.testing.assert_allclose(result.factor_prices, 1.0, atol=1e-9)


def test_measure_reports_every_solve():
    sam, model_spec = bench.national(5)
    rows = bench.measure("national-5", sam, model_spec, len(sam.accounts))
    assert [r["solve"] for r in rows] == ["base", "shock"]
    assert all(r["converged"] for r in rows)
    # The base year is its own start point: no root-finder evaluation at all.
    assert rows[0]["evaluations"] == 0 and rows[0]["base_gap"] < 1e-8
    assert rows[1]["evaluations"] > 0
