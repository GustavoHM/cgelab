"""Migration between regional factor pools, with origin income and destination supply."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions


def test_zero_migration_keeps_origin_allocation():
    _, econ = regions.build(regions.spec(factor_migration={"f_lab": 0.0}))
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-8
    migration = result.factor_pools["f_lab"]
    np.testing.assert_allclose(migration["supply"], np.eye(2) * 60.0, atol=1e-10)
    np.testing.assert_allclose(migration["endowment"], [60.0, 60.0], rtol=1e-10)
    assert set(migration["pools"]) == {"r1", "r2"}


def test_logit_migration_moves_workers_and_preserves_the_total():
    spec = regions.spec(factor_migration={"f_lab": 100.0}, factor_mobility_form="logit")
    proximity = pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], index=["r1", "r2"],
                             columns=["r1", "r2"])
    sam = regions.build()[0]
    econ = cge.calibrate(sam, spec, factor_pools={"f_lab": proximity})
    base = cge.solve_period(econ)
    shock = cge.solve_period(econ, scenario=cge.Scenario(productivity=np.array([0.8, 1.0])),
                             warm_start=base)
    assert shock.converged and abs(shock.walras) < 1e-8
    supply = shock.factor_pools["f_lab"]["supply"]
    initial = base.factor_pools["f_lab"]["supply"]
    assert supply[0, 0] < initial[0, 0]
    assert supply[1, 1] > initial[1, 1]
    np.testing.assert_allclose(supply.sum(axis=1), [60.0, 60.0], rtol=1e-8)
    np.testing.assert_allclose(supply.sum(), 120.0, rtol=1e-8)
    assert supply[0, 1] > 0.0 and supply[1, 0] > 0.0


def test_migration_is_optional_and_validated():
    _, econ = regions.build()
    assert cge.solve_period(econ).factor_pools is None
    with pytest.raises(KeyError, match="regional factors"):
        regions.build(regions.spec(factor_migration={"f_missing": 1.0}))
    with pytest.raises(ValueError, match=">= 0"):
        regions.build(regions.spec(factor_migration={"f_lab": -1.0}))
