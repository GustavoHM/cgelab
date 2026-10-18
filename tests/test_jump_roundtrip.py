"""Round trip through Julia: export, solve with Ipopt, compare with the nested solver.

Slow (Julia start-up and compilation), so it runs only when asked for:
``CGELAB_RUN_JULIA=1 pytest tests/test_jump_roundtrip.py``, with Julia on the PATH (or in
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

import cgelab as cge
from cgelab.examples import hosoe, toy

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
JULIA = os.environ.get("JULIA") or shutil.which("julia")

pytestmark = pytest.mark.skipif(
    os.environ.get("CGELAB_RUN_JULIA") != "1" or JULIA is None
    or not (VALIDATION / "Manifest.toml").exists(),
    reason="set CGELAB_RUN_JULIA=1, put julia on the PATH and instantiate validation/")


@pytest.fixture(scope="module")
def roundtrip():
    sys.path.insert(0, str(VALIDATION))
    os.environ.setdefault("JULIA", JULIA)
    import roundtrip_jump
    return roundtrip_jump


def test_cet_with_an_export_price_shock(roundtrip):
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet"))
    r = roundtrip.caso("pytest_cet", econ, cge.Closure(),
                       cge.Scenario(export_price=np.array([1.15, 0.9])))
    assert r["ok"], r


def test_institutions_with_fdi_and_a_tax_change(roundtrip):
    _, econ = toy.build_with_institutions()
    econ = dataclasses.replace(econ, hh_dtax=econ.hh_dtax * 1.5)
    r = roundtrip.caso("pytest_instituicoes", econ, cge.Closure(foreign="fixed_rate"),
                       cge.Scenario(fdi={"f_cap": 4.0}))
    assert r["ok"], r


def test_large_country_curves(roundtrip):
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet", export_demand=1.5,
                                            import_supply={"c_agr": 2.0}))
    r = roundtrip.caso("pytest_pais_grande", econ, cge.Closure(foreign="fixed_rate"),
                       cge.Scenario(export_price=np.array([1.15, 0.9]),
                                    world_price=np.array([1.1, 1.0])))
    assert r["ok"], r


def test_hosoe_tariff_abolition(roundtrip):
    _, econ = hosoe.build()
    r = roundtrip.caso("pytest_stdcge", econ, hosoe.CLOSURE,
                       cge.Scenario(tariff=np.zeros(2)))
    assert r["ok"], r


def test_imperfect_factor_mobility(roundtrip):
    for case, r in roundtrip.casos_mobilidade().items():
        assert r["ok"], (case, r)
