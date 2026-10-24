"""GTAP aggregation and a hand-balanced one-region SAM."""

import os
from pathlib import Path

import numpy as np
import pytest

import cgelab as cge
from cgelab.gtap import aggregate_gtap, gtap_sam, gtap_supply_use
from cgelab.har import Header


def header(name, array, sets, labels):
    array = np.asarray(array, dtype=float)
    array = array.reshape(array.shape + (1,) * (len(labels) - array.ndim))
    return Header(name, name, "RE", array, sets, labels)


def synthetic_gtap():
    from cgelab.examples import gtap as example

    return example.synthetic()


def test_one_region_supply_use_and_sam_close_hand_written_accounts():
    sut = gtap_supply_use(synthetic_gtap(), "r0")
    sut.check()
    assert sut.imports["c"] == 20.0
    assert sut.tariffs["c"] == 5.0
    assert sut.product_taxes["c"] == 5.0
    assert sut.production_taxes["a_c"] == 5.0
    assert sut.margins.loc["s", "c"] == 20.0
    assert sut.margins.loc["s", "s"] == -20.0
    sam = gtap_sam(synthetic_gtap(), "r0", name="synthetic GTAP")
    sam.check_balance(abs_tol=1e-8)
    assert sam.frame.value.sum() > 0


def test_one_region_gtap_sam_calibrates_and_solves_base_year():
    sam = gtap_sam(synthetic_gtap())
    spec = cge.ModelSpec(
        activities=["a_c", "a_s"], commodities=["c", "s"], households=["hh"],
        factors=["L"], government="gov", investment="s_i", rest_of_world="row",
        tariff_account="trf", margins=True, sigma_va=1.0)
    econ = cge.calibrate(sam, spec)
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-8
    np.testing.assert_allclose(result.X, [120.0, 20.0], atol=1e-7)
    np.testing.assert_allclose(result.Q, [120.0, 20.0], atol=1e-7)


def test_aggregation_sums_only_declared_values_by_axis_position():
    cube = header("CUBE", np.arange(1.0, 9.0).reshape(2, 2, 2),
                  ["COM", "REG", "REG"], [["c1", "c2"], ["r1", "r2"], ["r1", "r2"]])
    result = aggregate_gtap({"CUBE": cube},
                            {0: {"c1": "all", "c2": "all"},
                             1: {"r1": "all", "r2": "all"},
                             2: {"r1": "all", "r2": "all"}}, values=["CUBE"])
    assert result["CUBE"].array.shape == (1, 1, 1)
    assert result["CUBE"].array.item() == 36.0
    assert result["CUBE"].labels == [["all"], ["all"], ["all"]]


def test_aggregation_uses_explicit_weights_not_parameter_sums():
    value = header("ELAS", [[1.0, 2.0], [3.0, 4.0]], ["COM", "REG"],
                   [["c1", "c2"], ["r1", "r2"]])
    weight = header("WEIGHT", [[1.0, 1.0], [0.0, 2.0]], ["COM", "REG"],
                    [["c1", "c2"], ["r1", "r2"]])
    result = aggregate_gtap({"ELAS": value, "WEIGHT": weight},
                            {0: {"c1": "all", "c2": "all"},
                             1: {"r1": "all", "r2": "all"}},
                            values=[], weighted={"ELAS": (weight, ())})
    assert result["ELAS"].array.item() == pytest.approx(11.0 / 4.0)
    assert "WEIGHT" not in result
    zero = aggregate_gtap({"ELAS": value},
                          {0: {"c1": "all", "c2": "all"},
                           1: {"r1": "all", "r2": "all"}},
                          values=[], weighted={"ELAS": (np.zeros((2, 2)), ())})
    assert zero["ELAS"].array.item() == 0.0


def test_aggregation_and_region_validation_are_explicit():
    with pytest.raises(KeyError, match="MISSING"):
        aggregate_gtap({}, {0: {"c1": "all"}}, values=["MISSING"])
    with pytest.raises(KeyError, match="region 'r9'"):
        gtap_supply_use(synthetic_gtap(), "r9")


@pytest.mark.skipif(not os.environ.get("CGELAB_GTAP_DIR")
                    or not (Path(os.environ.get("CGELAB_GTAP_DIR", ".")) / "basedata.har").exists(),
                    reason="set CGELAB_GTAP_DIR to a GTAP 10 year folder")
def test_licensed_gtap_builds_and_solves_one_region():
    headers = cge.read_har(Path(os.environ["CGELAB_GTAP_DIR"]) / "basedata.har")
    sut = gtap_supply_use(headers, 0)
    sut.check(tol=1e-7)
    sam = gtap_sam(headers, 0, tol=1e-7)
    sam.check_balance(abs_tol=1e-3)
    spec = cge.ModelSpec(
        activities=sut.activities, commodities=sut.commodities, households=["hh"],
        factors=list(sut.value_added.index), government="gov", investment="s_i",
        rest_of_world="row", tariff_account="trf", margins=True)
    result = cge.solve_period(cge.calibrate(sam, spec))
    assert result.converged and abs(result.walras) < 1e-7
