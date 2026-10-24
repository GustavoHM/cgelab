"""SAM uncertainty: KL, hand balancing, a Beta--Leontief form, and model draws."""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy


def long(entries):
    return cge.SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]))


def beta_leontief_sam(a, final_demand=10.0):
    """One sector whose output d/(1-a) balances intermediate use and value added."""
    output = final_demand / (1.0 - a)
    return long([("x", "x", a * output), ("va", "x", (1.0 - a) * output),
                 ("x", "final", final_demand), ("final", "va", final_demand)])


def test_sam_distance_is_the_kl_written_by_hand():
    p = long([("a", "b", 10.0), ("b", "a", 20.0)])
    q = long([("a", "b", 20.0), ("b", "a", 30.0)])
    expected = np.log(5 / 6) / 3 + 2 * np.log(10 / 9) / 3
    assert cge.sam_distance(p, q) == pytest.approx(expected, rel=1e-14)
    assert cge.sam_distance(p, p) == 0.0
    assert cge.sam_distance(p, q) != cge.sam_distance(q, p)


def test_sam_distance_aligns_accounts_and_handles_support():
    p = long([("a", "b", 1.0), ("b", "a", 2.0)])
    q = long([("b", "a", 2.0), ("a", "b", 1.0), ("c", "c", 4.0)])
    assert cge.sam_distance(p, q) == pytest.approx(np.log(7 / 3), rel=1e-14)
    with pytest.raises(ValueError, match="negative"):
        cge.sam_distance(long([("a", "a", -1.0)]), p)


def test_dirichlet_leontief_multiplier_matches_the_beta_closed_form():
    a0, concentration = 0.2, 20.0
    original = beta_leontief_sam(a0)
    draws = cge.sam_draws(original, blocks={"production": (["x", "va"], ["x"])},
                          concentration=concentration, draws=5000, seed=17)
    multipliers = []
    for sam in draws.sams:
        sam.check_balance()
        mult = cge.accounting_multipliers(sam, endogenous=["x"])
        multipliers.append(mult.impact({"x": 1.0})["x"])
    alpha, beta = concentration * a0, concentration * (1.0 - a0)
    expected = (alpha + beta - 1.0) / (beta - 1.0)
    standard_error = np.std(multipliers, ddof=1) / np.sqrt(len(multipliers))
    assert np.mean(multipliers) == pytest.approx(expected, abs=6 * standard_error + 1e-12)
    assert draws.points.weight.sum() == pytest.approx(1.0)
    assert (draws.points.sam_distance >= 0).all()


def test_high_concentration_returns_to_the_original_and_preserves_zeros():
    original = beta_leontief_sam(0.2)
    before = original.frame.copy()
    draws = cge.sam_draws(original, blocks={"production": (["x", "va", "final"], ["x"])},
                          concentration=1e12, draws=3, seed=4)
    pd.testing.assert_frame_equal(original.frame, before)
    for sam in draws.sams:
        np.testing.assert_allclose(sam.matrix(original.accounts), original.matrix(),
                                   rtol=1e-5, atol=1e-5)
        assert ((sam.frame.row == "final") & (sam.frame.col == "x")).sum() == 0
    again = cge.sam_draws(original, blocks={"production": (["x", "va", "final"], ["x"])},
                          concentration=1e12, draws=3, seed=4)
    pd.testing.assert_frame_equal(draws.points, again.points)


def test_noise_draws_match_the_experiment_done_by_hand():
    sam, _ = toy.build()
    blocks = {"intermediate": (["c_agr", "c_ind"], ["a_agr", "a_ind"])}
    scale, seed = 0.05, 91
    draws = cge.sam_draws(sam, blocks=blocks, method="noise", scale=scale,
                          draws=2, seed=seed)
    rng = np.random.default_rng(seed)
    for expected in draws.sams:
        frame = sam.frame.copy(deep=True).reset_index(drop=True)
        selected = frame.index[frame.row.isin(blocks["intermediate"][0])
                               & frame.col.isin(blocks["intermediate"][1])
                               & (frame.value != 0)]
        frame.loc[selected, "value"] *= rng.lognormal(-0.5 * scale ** 2, scale,
                                                       size=len(selected))
        manual = cge.SAM.from_long(frame).balance()
        np.testing.assert_allclose(expected.matrix(sam.accounts), manual.matrix(sam.accounts),
                                   rtol=1e-12, atol=1e-10)
        expected.check_balance()


def test_negative_flows_can_use_noise_but_not_dirichlet_or_kl():
    sam = long([("a", "b", -10.0), ("b", "a", -10.0)])
    blocks = {"all": (["a", "b"], ["a", "b"])}
    with pytest.raises(ValueError, match="negative"):
        cge.sam_draws(sam, blocks=blocks, method="dirichlet", draws=1)
    draws = cge.sam_draws(sam, blocks=blocks, method="noise", draws=2)
    assert draws.points.sam_distance.isna().all()


def test_sam_sweep_recalibrates_solves_and_aligns_joint_draws():
    sam, _ = toy.build()
    draws = cge.sam_draws(sam, blocks={"intermediate": (["c_agr", "c_ind"],
                                                        ["a_agr", "a_ind"])},
                          concentration=50.0, draws=3, seed=23,
                          balance_kwargs={"rel_tol": 1e-12, "abs_tol": 1e-10})
    scenario = cge.Scenario(world_price=np.array([1.0, 1.2]))

    def measure(base, alt):
        return {"gdp": cge.real_gdp(alt)["expenditure"],
                "gdp_pct": 100 * (cge.real_gdp(alt)["expenditure"]
                                  / cge.real_gdp(base)["expenditure"] - 1)}

    result = cge.sam_sweep(draws, toy.spec(), measure=measure, scenario=scenario,
                           parameters={"sigma_va": cge.Uniform(0.6, 1.0)})
    assert isinstance(result, cge.SensitivityResult)
    assert result.method == "monte_carlo"
    assert result.outputs == ["gdp", "gdp_pct"]
    assert list(result.points.draw) == [0, 1, 2]
    assert result.points.sigma_va.between(0.6, 1.0).all()
    assert {"weight", "sam_distance", "sigma_va", "gdp", "gdp_pct"} <= set(result.points)
    assert result.summary().loc["gdp_pct", "mean"] == pytest.approx(result.points.gdp_pct.mean())
    for drawn in draws.sams:
        check = cge.base_year_reproduction(cge.calibrate(drawn, toy.spec()))
        assert check["ok"], check


def test_sam_sweep_names_the_failing_draw_and_checks_input():
    sam, _ = toy.build()
    draws = cge.sam_draws(sam, blocks={"intermediate": (["c_agr"], ["a_agr"])}, draws=2)

    def measure(base, alt):
        raise ValueError("changed base")

    with pytest.raises(RuntimeError, match="SAM draw 0"):
        cge.sam_sweep(draws, toy.spec(), measure=measure)
    with pytest.raises(TypeError, match="SAMDraws"):
        cge.sam_sweep(draws.sams, toy.spec(), measure=lambda b, a: 0.0)
    with pytest.raises(ValueError, match="overlap"):
        cge.sam_draws(sam, blocks={"one": (["c_agr"], ["a_agr"]),
                                   "two": (["c_agr"], ["a_agr"])}, draws=1)
    with pytest.raises(KeyError, match="not in the SAM"):
        cge.sam_draws(sam, blocks={"bad": (["missing"], ["a_agr"])}, draws=1)
    with pytest.raises(ValueError, match="concentration"):
        cge.sam_draws(sam, blocks={"one": (["c_agr"], ["a_agr"])},
                      concentration=0.0, draws=1)
