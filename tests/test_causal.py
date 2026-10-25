"""Treatment-minus-control contrasts computed from solved regions."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions, toy


def solved(shock=None):
    _, econ = regions.build()
    base = cge.solve_period(econ)
    return base, cge.solve_period(econ, scenario=shock, warm_start=base)


def test_no_change_has_zero_effect():
    base, alternative = solved()
    result = cge.treated_vs_control(base, alternative, treated="r1", controls="r2")
    assert result.loc["output", "effect"] == pytest.approx(0.0, abs=1e-12)
    assert result.loc["output", "treated_change"] == pytest.approx(0.0, abs=1e-12)


def test_treated_effect_matches_the_regional_changes_by_hand():
    tfp = np.ones(2)
    tfp[0] = 1.2
    base, alternative = solved(cge.Scenario(productivity=tfp))
    result = cge.treated_vs_control(base, alternative, treated="r1", controls="r2")
    treated_change = alternative.X[0] - base.X[0]
    control_change = alternative.X[1] - base.X[1]
    assert result.loc["output", "treated_change"] == pytest.approx(treated_change, rel=1e-10)
    assert result.loc["output", "control_change"] == pytest.approx(control_change, rel=1e-10)
    assert result.loc["output", "effect"] == pytest.approx(treated_change - control_change,
                                                            rel=1e-10)
    assert result.loc["output", "reference_treated"] == base.X[0]
    assert result.loc["output", "alternative_treated"] == alternative.X[0]


def test_a_common_symmetric_shock_cancels():
    base, alternative = solved(cge.Scenario(productivity=np.array([1.2, 1.2])))
    result = cge.treated_vs_control(base, alternative, treated="r1", controls="r2")
    assert result.loc["output", "effect"] == pytest.approx(0.0, abs=1e-8)


def test_multiple_outcomes_weights_and_a_callable():
    base, alternative = solved(cge.Scenario(transfers=np.array([-5.0, 5.0])))
    measures = {"income": "real_income",
                "r1_factor_price": lambda p, region: p.factor_prices[0]}
    result = cge.treated_vs_control(base, alternative, treated="r1", controls="r2",
                                    measure=measures, weights={"r2": 3.0})
    assert list(result.index) == ["income", "r1_factor_price"]
    change = (alternative.real["hh_real_income"][0] - base.real["hh_real_income"][0]
              - (alternative.real["hh_real_income"][1] - base.real["hh_real_income"][1]))
    assert result.loc["income", "effect"] == pytest.approx(change, rel=1e-10)
    assert result.loc["r1_factor_price", "effect"] == pytest.approx(0.0, abs=1e-12)


def test_bilateral_outflows_use_the_origin_region():
    tfp = np.ones(2)
    tfp[0] = 1.2
    base, alternative = solved(cge.Scenario(productivity=tfp))
    result = cge.treated_vs_control(base, alternative, treated="r1", controls="r2",
                                    measure="bilateral")
    expected = (alternative.bilateral[0].sum() - base.bilateral[0].sum()
                - (alternative.bilateral[1].sum() - base.bilateral[1].sum()))
    assert result.loc["bilateral", "effect"] == pytest.approx(expected, rel=1e-10)


def test_invalid_regions_weights_and_models_are_refused():
    base, alternative = solved()
    with pytest.raises(ValueError, match="controls"):
        cge.treated_vs_control(base, alternative, treated="r1", controls="r1")
    with pytest.raises(KeyError, match="r3"):
        cge.treated_vs_control(base, alternative, treated="r1", controls="r3")
    with pytest.raises(ValueError, match="every control"):
        cge.treated_vs_control(base, alternative, treated="r1", controls="r2",
                               weights={"other": 1.0})
    with pytest.raises(ValueError, match="positive sum"):
        cge.treated_vs_control(base, alternative, treated="r1", controls="r2",
                               weights={"r2": 0.0})
    with pytest.raises(ValueError, match="unknown regional measure"):
        cge.treated_vs_control(base, alternative, treated="r1", controls="r2",
                               measure="unknown")
    _, national = toy.build()
    period = cge.solve_period(national)
    with pytest.raises(ValueError, match="regional"):
        cge.treated_vs_control(period, period, treated="r1", controls="r2")
