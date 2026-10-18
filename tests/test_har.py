"""GEMPACK header array files: round trips, and the GTAP identities when the data are here.

The GTAP database is licensed and is not in the repository. Point
``CGELAB_GTAP_DIR`` at one year of a GTAP 10 package (the folder with
``basedata.har``) to run the identity checks.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import cgelab as cge
from cgelab.har import Header

GTAP = os.environ.get("CGELAB_GTAP_DIR")


def _sample():
    cube = np.random.default_rng(0).normal(size=(3, 4, 4)).astype(np.float32).astype(float)
    return {
        "SETA": np.array(["alpha", "beta", "gamma"], dtype=object),
        "CUBE": Header("CUBE", "a labelled cube", "RE", cube, ["COM", "REG", "REG"],
                       [["c1", "c2", "c3"], ["r1", "r2", "r3", "r4"], ["r1", "r2", "r3", "r4"]]),
        "FLAG": np.array([[1], [0], [1]], dtype=np.int64),
        "PLAN": np.arange(6, dtype=float).reshape(2, 3),
        "BLOC": np.arange(24, dtype=float).reshape(2, 3, 4),
    }


def test_round_trip(tmp_path):
    src = _sample()
    cge.write_har(tmp_path / "x.har", src)
    got = cge.read_har(tmp_path / "x.har")
    assert list(got) == list(src)
    assert list(got["SETA"].array) == ["alpha", "beta", "gamma"]
    cube = got["CUBE"]
    assert cube.kind == "RE" and cube.sets == ["COM", "REG", "REG"]
    assert cube.labels[2] == ["r1", "r2", "r3", "r4"]
    np.testing.assert_array_equal(cube.array, src["CUBE"].array)
    assert cube.description == "a labelled cube"
    np.testing.assert_array_equal(got["FLAG"].array, src["FLAG"])
    assert got["PLAN"].kind == "2R"
    np.testing.assert_array_equal(got["PLAN"].array, src["PLAN"])
    assert got["BLOC"].kind == "RL"
    np.testing.assert_array_equal(got["BLOC"].array, src["BLOC"])


def test_single_precision_is_what_the_format_holds(tmp_path):
    cge.write_har(tmp_path / "p.har", {"PI  ": np.array([[np.pi]])})
    value = cge.read_har(tmp_path / "p.har")["PI"].array[0, 0]
    assert value == pytest.approx(np.pi, rel=1e-7) and value != np.pi


def test_labelled_long_form(tmp_path):
    cge.write_har(tmp_path / "x.har", _sample())
    s = cge.read_har(tmp_path / "x.har", headers=["CUBE"])["CUBE"].to_series()
    assert list(s.index.names) == ["COM", "REG", "REG"]
    assert s.loc[("c2", "r3", "r4")] == pytest.approx(_sample()["CUBE"].array[1, 2, 3])


def test_damage_is_refused(tmp_path):
    cge.write_har(tmp_path / "x.har", _sample())
    raw = bytearray((tmp_path / "x.har").read_bytes())
    raw[8:12] = b"\x00\x00\x00\x09"            # the first record's trailer
    (tmp_path / "bad.har").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="trailer|truncated"):
        cge.read_har(tmp_path / "bad.har")
    with pytest.raises(ValueError, match="four characters"):
        cge.write_har(tmp_path / "y.har", {"TOOLONG": np.zeros((1, 1))})


gtap = pytest.mark.skipif(not GTAP or not (Path(GTAP or ".") / "basedata.har").exists(),
                          reason="set CGELAB_GTAP_DIR to a GTAP 10 year folder")


@pytest.fixture(scope="module")
def base():
    return cge.read_har(Path(GTAP) / "basedata.har")


@gtap
def test_gtap_cif_is_fob_plus_margins(base):
    A = lambda k: base[k].array  # noqa: E731
    gap = A("VIWS") - A("VXWD") - A("VTWR").sum(axis=0)
    assert np.abs(gap).max() < 1e-6 * A("VIWS").max()


@gtap
def test_gtap_margin_supply_meets_demand(base):
    np.testing.assert_allclose(base["VST"].array.sum(axis=1),
                               base["VTWR"].array.sum(axis=(1, 2, 3)), rtol=1e-5)


@gtap
def test_gtap_zero_profit(base):
    A = lambda k: base[k].array  # noqa: E731
    cost = (A("VDFA").sum(0) + A("VIFA").sum(0) + A("EVFA").sum(0))[:A("TVOM").shape[0]]
    revenue = A("TVOM") + A("OSEP")
    assert np.abs(cost - revenue).max() < 1e-6 * revenue.max()


@gtap
def test_gtap_output_is_sold(base):
    A = lambda k: base[k].array  # noqa: E731
    tv = A("TVOM")
    sales = A("VDFM").sum(1) + A("VDPM") + A("VDGM") + A("VXMD").sum(2)
    marg = np.zeros_like(tv)
    marg[[base["TVOM"].labels[0].index(m) for m in base["VST"].labels[0]]] = A("VST")
    assert np.abs(tv - sales - marg).max() < 1e-6 * tv.max()
