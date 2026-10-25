"""Optional regional government and investment accounts."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions


def build_institutional():
    rows = []
    for r, c, value in regions.ENTRIES:
        origin, destination = r.split(":")[0], c.split(":")[0]
        if r == "gov":
            r = f"{destination}:gov"
        if c == "gov":
            c = f"{origin}:gov"
        if r == "s_i":
            r = f"{destination}:s_i"
        if c == "s_i":
            c = f"{origin}:s_i"
        rows.append((r, c, value))
    sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    sam.check_balance(abs_tol=1e-9)
    spec = regions.spec(government="r1:gov", investment="r1:s_i",
                        regional_government={"r2": "r2:gov"},
                        regional_investment={"r2": "r2:s_i"})
    return cge.calibrate(sam, spec)


def test_institution_accounts_are_mapped_and_solved():
    econ = build_institutional()
    assert econ.regional.government_accounts == {"r2": "r2:gov"}
    assert econ.regional.investment_accounts == {"r2": "r2:s_i"}
    result = cge.solve_period(econ)
    assert result.converged and abs(result.walras) < 1e-8
    assert result.real["gov_real"] == pytest.approx(20.0, rel=1e-10)
    assert result.real["inv_real"] == pytest.approx(40.0, rel=1e-10)
    expected = [[[45.0], [18.0]], [[18.0], [72.0]]]
    expected_sam = [[[50.0], [20.0]], [[20.0], [80.0]]]
    np.testing.assert_allclose(result.bilateral, expected, rtol=1e-10)
    np.testing.assert_allclose(result.bilateral + result.bilateral_imports,
                               expected_sam, rtol=1e-10)


def test_fixed_real_investment_includes_regional_accounts():
    econ = build_institutional()
    result = cge.solve_period(econ, cge.Closure(investment="fixed_real"),
                              cge.Scenario(investment_level=80.0))
    assert result.converged and abs(result.walras) < 1e-8
    assert result.real["inv_real"] == pytest.approx(80.0, rel=1e-10)
    financed = (result.household_saving + result.forced_saving
                + result.real["government_saving"] + result.foreign_saving)
    assert financed == pytest.approx(result.savings, rel=1e-10)


def test_invalid_institution_maps_are_refused():
    with pytest.raises(ValueError, match="more than one region"):
        build = regions.spec(regional_government={"r1": "gov", "r2": "gov"})
        rows = [(r, c, v) for r, c, v in regions.ENTRIES]
        sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
        cge.calibrate(sam, build)
    with pytest.raises(ValueError, match="regions not in the model"):
        build = regions.spec(regional_investment={"r3": "s_i"})
        rows = [(r, c, v) for r, c, v in regions.ENTRIES]
        sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
        cge.calibrate(sam, build)
