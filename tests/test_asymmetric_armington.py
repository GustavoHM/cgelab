"""Asymmetric Armington: one elasticity where imports get dearer, another where they get cheaper."""

import dataclasses

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy

SAM = toy._frame(toy._BASE)
LOW, HIGH = 0.9, 3.0


def _econ(**kw):
    return cge.calibrate(SAM, toy.spec(export_rule="cet", sigma_armington=LOW,
                                       sigma_armington_falling=HIGH, **kw))


ECON = _econ()


def test_the_base_year_reproduces_with_either_branch():
    period = cge.solve_period(ECON)
    assert period.converged and abs(period.walras) < 1e-12
    np.testing.assert_allclose(period.imports, ECON.imp0, atol=1e-12)


@pytest.mark.parametrize("world", [[1.0, 1.25], [1.0, 0.75], [1.3, 0.8], [0.7, 0.75]])
def test_each_commodity_is_the_symmetric_model_at_the_elasticity_its_regime_selects(world):
    """The known solution: fix each commodity at the elasticity selected, solve symmetric."""
    sc = cge.Scenario(world_price=np.array(world))
    asym = cge.solve_period(ECON, scenario=sc)
    assert asym.converged and abs(asym.walras) < 1e-12
    # The regime follows the relative price: low where imports got dearer.
    pm = asym.exchange_rate * np.array(world)
    dearer = pm > asym.p_domestic
    np.testing.assert_array_equal(asym.armington_sigma, np.where(dearer, LOW, HIGH))
    sym_econ = cge.calibrate(SAM, toy.spec(
        export_rule="cet", sigma_armington=dict(zip(ECON.coms, asym.armington_sigma))))
    sym = cge.solve_period(sym_econ, scenario=sc)
    assert sym.armington_sigma is None
    for field in ("factor_prices", "imports", "exports", "pq", "hh_income"):
        np.testing.assert_allclose(getattr(asym, field), getattr(sym, field), rtol=1e-11)
    assert asym.exchange_rate == pytest.approx(sym.exchange_rate, rel=1e-11)


def test_a_dearer_import_is_substituted_less_than_under_the_high_elasticity():
    sc = cge.Scenario(world_price=np.array([1.0, 1.25]))
    asym = cge.solve_period(ECON, scenario=sc)
    high = cge.solve_period(cge.calibrate(SAM, toy.spec(export_rule="cet",
                                                        sigma_armington=HIGH)), scenario=sc)
    i = ECON.commodity("c_ind")
    assert asym.armington_sigma[i] == LOW
    assert asym.imports[i] > high.imports[i]


def test_equal_elasticities_are_the_symmetric_model():
    same = cge.calibrate(SAM, toy.spec(
        export_rule="cet", sigma_armington=2.0, sigma_armington_falling=2.0))
    plain = cge.calibrate(SAM, toy.spec(export_rule="cet", sigma_armington=2.0))
    sc = cge.Scenario(world_price=np.array([1.2, 0.9]))
    a, b = cge.solve_period(same, scenario=sc), cge.solve_period(plain, scenario=sc)
    np.testing.assert_allclose(a.imports, b.imports, rtol=1e-12)


def test_a_commodity_left_out_of_the_mapping_keeps_one_elasticity():
    econ = cge.calibrate(SAM, toy.spec(
        export_rule="cet", sigma_armington=LOW, sigma_armington_falling={"c_ind": HIGH}))
    assert np.isnan(econ.sigma_arm_falling[econ.commodity("c_agr")])
    r = cge.solve_period(econ, scenario=cge.Scenario(world_price=np.array([0.7, 0.75])))
    assert r.armington_sigma[econ.commodity("c_agr")] == LOW
    assert r.armington_sigma[econ.commodity("c_ind")] == HIGH


def test_without_it_nothing_changes():
    _, plain = toy.build()
    assert plain.sigma_arm_falling is None
    assert cge.solve_period(plain).armington_sigma is None


def test_a_falling_elasticity_below_the_rising_one_is_refused():
    with pytest.raises(ValueError, match="below sigma_armington"):
        cge.calibrate(SAM, toy.spec(sigma_armington=2.0, sigma_armington_falling=1.0))


def test_the_replica_and_the_backends_refuse_it():
    with pytest.raises(ValueError, match="asymmetric Armington"):
        cge.to_jump(ECON)


def test_the_regional_preflight_lists_it():
    from cgelab.regional import _slice_issues
    econ = dataclasses.replace(ECON, spec=dataclasses.replace(ECON.spec, regions=("r1",)))
    assert ("asymmetric Armington", "ModelSpec.sigma_armington_falling", "") in \
        _slice_issues(econ, SAM)
