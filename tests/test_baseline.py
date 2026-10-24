"""A recursive baseline calibrated to a path of real GDP.

The checks are inverse problems: a path solved with a productivity that is known
gives a GDP, and calibrating on that GDP, with the productivity left out, has to
give the productivity back. One of them is checked against the Solow recursion
worked on paper.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy

YEARS = list(range(2021, 2033))
T = np.arange(len(YEARS))


def gdp_of(path, measure="expenditure"):
    if measure == "expenditure":
        return pd.Series([cge.real_gdp(p)["expenditure"] for p in path.periods], index=YEARS)
    return pd.Series([cge.real_value_added(p).sum() for p in path.periods], index=YEARS)


def test_real_value_added_at_the_base_year_is_the_factor_bill_and_the_activity_tax():
    _, econ = toy.build()
    base = cge.solve_period(econ)
    va = cge.real_value_added(base)
    g = cge.real_gdp(base)
    assert va.sum() == pytest.approx(g["value_added"] + g["activity_tax"], rel=1e-12)
    # Productivity moves it; the factor bill of real_gdp does not move.
    up = cge.solve_period(econ, scenario=cge.Scenario(productivity=np.full(2, 1.1)))
    assert cge.real_gdp(up)["value_added"] == pytest.approx(g["value_added"], rel=1e-12)
    assert cge.real_value_added(up).sum() > 1.1 * va.sum()


@pytest.mark.parametrize("kind", ["growth", "index", "level"])
@pytest.mark.parametrize("measure", ["expenditure", "production"])
def test_calibrating_on_a_path_gives_its_productivity_back(kind, measure):
    _, econ = toy.build()
    known = cge.run_path(econ, years=YEARS, growth=cge.Growth(tfp=0.02, labour=0.01))
    level = gdp_of(known, measure)
    gdp = {"growth": level.pct_change(), "index": level / level.iloc[0] * 7.0,
           "level": level}[kind]
    b = cge.calibrate_baseline(econ, years=YEARS, gdp=gdp, kind=kind, measure=measure,
                               growth=cge.Growth(labour=0.01))
    assert np.allclose(b.factor.to_numpy(), 1.02 ** T, rtol=1e-10, atol=0)
    assert np.allclose(b.productivity.to_numpy(), (1.02 ** T)[:, None], rtol=1e-10, atol=0)
    assert np.allclose(b.achieved, level, rtol=1e-10, atol=0)
    assert np.allclose(b.path.capital_stock, known.capital_stock, rtol=1e-10, atol=0)
    assert b.path.converged and b.path.worst_walras < 1e-9


def test_the_solow_recursion_worked_on_paper():
    """One good, Cobb–Douglas, closed, saving a fifth of income.

    Labour 60 and capital services 40 make 100, so ``Y = A 100 (S/40)^0.4`` with
    the labour force fixed. Services are the stock times the user cost of 0.10,
    the stock starts at 400, and ``K' = 0.95 K + 0.2 Y``. Given ``Y``, each
    year's ``A`` follows from the stock that year, and the stock from the ``Y``
    of the years before.
    """
    _, econ = toy.build_one_good()
    target = pd.Series(100.0 * 1.03 ** T * (1.0 + 0.02 * np.sin(T)), index=YEARS)
    b = cge.calibrate_baseline(econ, years=YEARS, gdp=target, kind="level")
    k, a = 400.0, []
    for y in target:
        a.append(y / (100.0 * (0.10 * k / 40.0) ** 0.4))
        k = 0.95 * k + 0.2 * y
    assert np.allclose(b.factor.to_numpy(), a, rtol=1e-10, atol=0)
    assert np.allclose(b.achieved, target, rtol=1e-10, atol=0)


def test_a_pattern_shares_the_year_unknown_among_activities():
    _, econ = toy.build()
    lam = 1.01 ** T

    def known_sc(year, t):
        return cge.Scenario(productivity=np.array([lam[t] ** 2, lam[t]]))

    known = cge.run_path(econ, years=YEARS, scenario=known_sc)
    b = cge.calibrate_baseline(econ, years=YEARS, gdp=gdp_of(known).pct_change(),
                               pattern={"a_agr": 2.0, "a_ind": 1.0})
    assert np.allclose(b.factor.to_numpy(), lam, rtol=1e-10, atol=0)
    assert np.allclose(b.productivity["a_agr"], lam ** 2, rtol=1e-10, atol=0)


@pytest.mark.parametrize("kind", ["growth", "level"])
def test_sectoral_value_added_gives_each_activity_its_own_productivity(kind):
    _, econ = toy.build()
    rates = np.array([0.03, 0.01])

    def known_sc(year, t):
        return cge.Scenario(productivity=(1.0 + rates) ** t)

    known = cge.run_path(econ, years=YEARS, scenario=known_sc, growth=cge.Growth(labour=0.01))
    va = pd.DataFrame([cge.real_value_added(p) for p in known.periods], index=YEARS)
    sectors = va.pct_change() if kind == "growth" else va
    b = cge.calibrate_baseline(econ, years=YEARS, sectors=sectors, kind=kind,
                               growth=cge.Growth(labour=0.01))
    assert list(b.factor.columns) == ["a_agr", "a_ind"]
    assert np.allclose(b.factor.to_numpy(), (1.0 + rates) ** T[:, None], rtol=1e-9, atol=0)
    assert np.allclose(b.achieved.to_numpy(), va.to_numpy(), rtol=1e-10, atol=0)


def test_a_scenario_productivity_stays_under_the_calibrated_one():
    """What the scenario sets is kept; the calibration finds what it leaves to explain."""
    _, econ = toy.build()
    known = cge.run_path(econ, years=YEARS, growth=cge.Growth(tfp=0.02))
    own = np.array([1.05, 1.0])
    b = cge.calibrate_baseline(econ, years=YEARS, gdp=gdp_of(known), kind="level",
                               scenario=cge.Scenario(productivity=own))
    assert np.allclose(b.achieved, gdp_of(known), rtol=1e-10, atol=0)
    assert np.allclose(b.productivity.to_numpy(), b.multiplier.to_numpy() * own, rtol=1e-14)
    assert (b.factor < 1.02 ** T).iloc[1:].all()      # the 5% in farming explains part of it


def test_the_null_policy_reproduces_the_baseline_and_a_policy_moves_gdp():
    _, econ = toy.build()
    target = pd.Series(0.025, index=YEARS)
    growth = cge.Growth(labour=0.01)
    b = cge.calibrate_baseline(econ, years=YEARS, gdp=target, growth=growth)
    same = cge.run_path(econ, years=YEARS, scenario=b.scenario(), growth=growth)
    assert np.allclose(gdp_of(same), b.achieved, rtol=1e-12, atol=0)
    assert np.allclose(same.capital_stock, b.path.capital_stock, rtol=1e-12, atol=0)
    assert np.allclose(b.run().capital_stock, b.path.capital_stock, rtol=1e-12, atol=0)

    tax = cge.Scenario(product_tax=econ.ptax * 1.5)
    policy = b.run(tax)
    moved = gdp_of(policy) / b.achieved - 1.0
    assert (moved.abs() > 1e-6).all()
    for t, p in enumerate(policy.periods):           # productivity stays the calibrated one
        assert np.allclose(p.scenario.productivity, b.multiplier.iloc[t], rtol=1e-14)
    # A productivity the policy sets multiplies the calibrated one.
    boost = b.run(cge.Scenario(productivity=np.array([1.1, 1.0])))
    assert np.allclose(boost.periods[3].scenario.productivity,
                       b.multiplier.iloc[3] * [1.1, 1.0], rtol=1e-14)
    with pytest.raises(ValueError, match="2040"):
        b.scenario()(2040, 19)


def test_what_calibrate_baseline_refuses():
    _, econ = toy.build()
    target = pd.Series(0.02, index=YEARS)
    with pytest.raises(ValueError, match=r"\[2032\]"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target.iloc[:-1])
    with pytest.raises(ValueError, match="every year"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target.iloc[1:], kind="level")
    with pytest.raises(ValueError, match="one of them"):
        cge.calibrate_baseline(econ, years=YEARS)
    with pytest.raises(ValueError, match="one of them"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target,
                               sectors=pd.DataFrame({"a_agr": target}))
    with pytest.raises(ValueError, match="a_svc"):
        cge.calibrate_baseline(econ, years=YEARS, sectors=pd.DataFrame({"a_svc": target}))
    with pytest.raises(ValueError, match="a_svc"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target, pattern={"a_svc": 1.0})
    with pytest.raises(ValueError, match="not all zero"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target, pattern=[0.0, 0.0])
    with pytest.raises(ValueError, match="kind"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target, kind="rate")
    with pytest.raises(ValueError, match="measure"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target, measure="income")
    with pytest.raises(ValueError, match="2023"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target.where(target.index != 2023, -1.5))


def test_a_target_no_productivity_reaches_names_the_year():
    """Farming alone carries the unknown, so industry's output is a floor GDP cannot go under."""
    _, econ = toy.build()
    base = cge.real_gdp(cge.solve_period(econ))["expenditure"]
    target = pd.Series(base, index=YEARS)
    target[2025] = 0.3 * base
    with pytest.raises(ValueError, match="2025"):
        cge.calibrate_baseline(econ, years=YEARS, gdp=target, kind="level",
                               pattern={"a_agr": 1.0})
