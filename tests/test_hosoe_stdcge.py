"""Tariff abolition in Hosoe, Gasawa and Hashimoto's stdcge and lrgcge.

The reference is `validation/hosoe.jl`, a line-by-line JuMP transcription of the
two GAMS library programs, solved with Ipopt and stored by
`validation/replicate_hosoe.py --write`. Every variable the programs display is
compared, at the base year and with every tariff set to zero.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import hosoe

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "hosoe_tariff_abolition.json").read_text())
CASES = {"base": None, "abolicao": ck.Scenario(tariff=np.zeros(2))}


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", float(v)


@pytest.mark.parametrize("model,large", [("stdcge", False), ("lrgcge", True)])
@pytest.mark.parametrize("case", list(CASES))
def test_every_displayed_variable_matches_the_transcription(model, large, case):
    _, econ = hosoe.build(large_country=large)
    res = ck.solve_period(econ, hosoe.CLOSURE, CASES[case])
    assert res.converged and abs(res.walras) < 1e-10
    ours = dict(_flat(hosoe.variables(econ, res)))
    ref = dict(_flat(REFERENCE[model][case]))
    assert set(ours) == set(ref)
    for name, value in ref.items():
        assert ours[name] == pytest.approx(value, rel=1e-9, abs=1e-9), name


def test_the_calibration_is_hosoes():
    _, econ = hosoe.build()
    np.testing.assert_allclose(econ.tm0, [1 / 13, 2 / 11])
    np.testing.assert_allclose(econ.tauz0, [5 / 73, 4 / 72])
    np.testing.assert_allclose(econ.ptax, 0.0, atol=1e-15)    # both taxes are in the prices
    np.testing.assert_allclose(econ.dom0, [70.0, 72 + 4 - 4])  # Hosoe's D0 = (1+tauz)Z0 - E0
    assert econ.gov_savrate == pytest.approx(2 / 35)


def test_abolition_removes_the_whole_tariff():
    _, econ = hosoe.build()
    out = hosoe.variables(econ, ck.solve_period(econ, hosoe.CLOSURE, CASES["abolicao"]))
    assert out["Tm"] == {"BRD": 0.0, "MLK": 0.0}
    base = REFERENCE["stdcge"]["base"]
    # Total imports rise; milk, which carried the higher tariff, takes the share.
    assert sum(out["M"].values()) > sum(base["M"].values())
    assert out["M"]["MLK"] / out["M"]["BRD"] > base["M"]["MLK"] / base["M"]["BRD"]
    assert out["epsilon"] > 1.0           # the currency depreciates to pay for the imports
