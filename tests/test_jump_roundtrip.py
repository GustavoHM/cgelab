"""Round trip through Julia: export, solve with Ipopt, compare with the nested solver.

Slow (Julia start-up and compilation), so it runs only when asked for:
``CGEKIT_RUN_JULIA=1 pytest tests/test_jump_roundtrip.py``, with Julia on the PATH (or in
``JULIA``) and the environment in ``validation/`` instantiated. The full set of cases is
``validation/roundtrip_jump.py``.
"""

import dataclasses
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import cgekit as ck
from cgekit.examples import toy

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
JULIA = os.environ.get("JULIA") or shutil.which("julia")

pytestmark = pytest.mark.skipif(
    os.environ.get("CGEKIT_RUN_JULIA") != "1" or JULIA is None
    or not (VALIDATION / "Manifest.toml").exists(),
    reason="set CGEKIT_RUN_JULIA=1, put julia on the PATH and instantiate validation/")


@pytest.fixture(scope="module")
def roundtrip():
    sys.path.insert(0, str(VALIDATION))
    os.environ.setdefault("JULIA", JULIA)
    import roundtrip_jump
    return roundtrip_jump


def test_cet_with_an_export_price_shock(roundtrip):
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet"))
    r = roundtrip.caso("pytest_cet", econ, ck.Closure(),
                       ck.Scenario(export_price=np.array([1.15, 0.9])))
    assert r["ok"], r


def test_institutions_with_fdi_and_a_tax_change(roundtrip):
    _, econ = toy.build_with_institutions()
    econ = dataclasses.replace(econ, hh_dtax=econ.hh_dtax * 1.5)
    r = roundtrip.caso("pytest_instituicoes", econ, ck.Closure(foreign="fixed_rate"),
                       ck.Scenario(fdi={"f_cap": 4.0}))
    assert r["ok"], r
