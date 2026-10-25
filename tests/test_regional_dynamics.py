"""Regional capital accumulation and the simulated panel."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def regional_path():
    _, econ = regions.build()
    growth = cge.Growth(labour=0.01, depreciation=0.01,
                        labour_factor="r1:f_lab", capital_factor=["r1:f_cap", "r2:f_cap"])
    return cge.run_path(econ, years=[2021, 2022], growth=growth)


def test_capital_accumulates_by_region():
    path = regional_path()
    assert path.converged
    assert path.capital_stock.shape == (2, 2)
    assert (path.capital_stock[1] > path.capital_stock[0]).all()
    assert (path[2022].X > path[2021].X).all()
    assert abs(path.worst_walras) < 1e-8


def test_simulated_panel_is_tidy_and_matches_solved_values():
    path = regional_path()
    panel = cge.simulated_panel(path, treated="r1")
    assert list(panel.columns) == ["year", "region", "outcome", "value", "treated", "noise"]
    assert len(panel) == 4
    assert set(panel.year) == {2021, 2022}
    assert set(panel.region) == {"r1", "r2"}
    assert set(panel.outcome) == {"output"}
    assert panel.treated.tolist() == [True, False, True, False]
    assert (panel.noise == 1.0).all()
    for year in path.years:
        for region in path.econ.regional.regions:
            i = path.econ.regional.regions.index(region)
            expected = path[year].X[path.econ.regional.activity_region == i].sum()
            actual = panel.query("year == @year and region == @region").value.iloc[0]
            assert actual == pytest.approx(expected, rel=1e-10)


def test_noise_is_seeded_and_does_not_change_the_path():
    path = regional_path()
    first = cge.simulated_panel(path, noise=0.1, seed=17)
    second = cge.simulated_panel(path, noise=0.1, seed=17)
    third = cge.simulated_panel(path, noise=0.1, seed=18)
    assert first.equals(second)
    assert not np.allclose(first.value, third.value)
    assert (first.noise > 0).all()
    assert (third.noise > 0).all()


def test_panel_validation():
    path = regional_path()
    with pytest.raises(ValueError, match="nonnegative"):
        cge.simulated_panel(path, noise=-0.1)
    with pytest.raises(KeyError, match="not in the model"):
        cge.simulated_panel(path, treated="r3")
    _, national = toy.build()
    national_path = cge.run_path(national, years=[2021])
    with pytest.raises(ValueError, match="regional"):
        cge.simulated_panel(national_path)
    with pytest.raises(ValueError, match="full 'region:factor'"):
        cge.run_path(regions.build()[1], years=[2021], growth=cge.Growth())
