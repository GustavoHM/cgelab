"""A synthetic GTAP-shaped dataset with hand-balanced accounts."""

from __future__ import annotations

import numpy as np

from ..calibrate import calibrate
from ..gtap import gtap_sam
from ..har import Header
from ..spec import ModelSpec

__all__ = ["synthetic", "build"]


def _header(name, array, sets, labels):
    array = np.asarray(array, dtype=float)
    array = array.reshape(array.shape + (1,) * (len(labels) - array.ndim))
    return Header(name, name, "RE", array, sets, labels)


def synthetic():
    """Two commodities, one factor, one region and one transport mode."""
    data = {
        "TVOM": _header("TVOM", [120.0, 20.0], ["PROD", "REG"], [["c", "s"], ["r0"]]),
        "OSEP": _header("OSEP", [5.0, 0.0], ["PROD", "REG"], [["c", "s"], ["r0"]]),
        "VDFM": _header("VDFM", np.array([10.0, 5.0, 0.0, 0.0]).reshape(2, 2, 1),
                        ["COM", "ACT", "REG"], [["c", "s"], ["c", "s"], ["r0"]]),
        "VIFM": _header("VIFM", np.zeros((2, 2, 1)),
                        ["COM", "ACT", "REG"], [["c", "s"], ["c", "s"], ["r0"]]),
        "EVFA": _header("EVFA", [[105.0, 15.0]], ["FAC", "ACT", "REG"],
                        [["L"], ["c", "s"], ["r0"]]),
        "VDPM": _header("VDPM", [85.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VIPM": _header("VIPM", [30.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VPA": _header("VPA", [120.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VDGM": _header("VDGM", [10.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VIGM": _header("VIGM", [0.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VGA": _header("VGA", [10.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VINV": _header("VINV", [10.0, 0.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
        "VXMD": _header("VXMD", np.array([15.0, 0.0]).reshape(2, 1, 1),
                        ["COM", "REG", "REG"], [["c", "s"], ["r0"], ["r0"]]),
        "VXWD": _header("VXWD", np.array([20.0, 0.0]).reshape(2, 1, 1),
                        ["COM", "REG", "REG"], [["c", "s"], ["r0"], ["r0"]]),
        "VIWS": _header("VIWS", np.array([40.0, 0.0]).reshape(2, 1, 1),
                        ["COM", "REG", "REG"], [["c", "s"], ["r0"], ["r0"]]),
        "VIMS": _header("VIMS", np.array([45.0, 0.0]).reshape(2, 1, 1),
                        ["COM", "REG", "REG"], [["c", "s"], ["r0"], ["r0"]]),
        "VTWR": _header("VTWR", np.array([20.0, 0.0]).reshape(1, 2, 1, 1),
                        ["MODE", "COM", "REG", "REG"], [["s"], ["c", "s"], ["r0"], ["r0"]]),
        "VST": _header("VST", [0.0, 20.0], ["COM", "REG"], [["c", "s"], ["r0"]]),
    }
    data["VDFA"] = data["VDFM"]
    data["VIFA"] = data["VIFM"]
    return data


def build():
    """Return the synthetic GTAP SAM and its calibrated one-region economy."""
    sam = gtap_sam(synthetic())
    spec = ModelSpec(
        activities=["a_c", "a_s"], commodities=["c", "s"], households=["hh"],
        factors=["L"], government="gov", investment="s_i", rest_of_world="row",
        tariff_account="trf", margins=True, sigma_va=1.0)
    return sam, calibrate(sam, spec)
