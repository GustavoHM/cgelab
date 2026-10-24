"""Minimum-distance matching: weighted least squares and a CGE recovery."""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy
from cgelab.match import _match


def test_linear_match_is_the_weighted_least_squares_solution():
    A = np.array([[2.0, 1.0], [1.0, -1.0]])
    target = np.array([2.0, 0.0])
    covariance = np.diag([1.0, 4.0])

    def run(x):
        return {"b1": 2 * x["x"] + x["y"], "b2": x["x"] - x["y"]}

    opt, best, achieved, residuals, evaluations = _match(
        run, {"b1": target[0], "b2": target[1]}, {"x": (-10, 10), "y": (-10, 10)},
        covariance=covariance)
    expected = np.linalg.solve(A.T @ np.linalg.inv(covariance) @ A,
                               A.T @ np.linalg.inv(covariance) @ target)
    assert opt.success
    assert [best["x"], best["y"]] == pytest.approx(expected, rel=1e-7)
    assert best["distance"] == pytest.approx(0.0, abs=1e-14)
    assert list(achieved) == ["b1", "b2"]
    assert residuals["b1"] == pytest.approx(0.0, abs=1e-8)
    assert len(evaluations) >= 2


def test_singular_covariance_uses_a_pseudo_inverse():
    def run(x):
        return {"b1": x["x"], "b2": 2 * x["x"]}

    opt, best, achieved, _, _ = _match(run, {"b1": 1.0, "b2": 2.0},
                                        {"x": (0, 2)}, covariance=np.ones((2, 2)))
    assert opt.success
    assert best["x"] == pytest.approx(1.0, rel=1e-8)
    assert best["distance"] == pytest.approx(0.0, abs=1e-14)


def test_bounds_and_optimizer_failure_are_reported():
    opt, best, achieved, residuals, _ = _match(lambda x: x["x"], {"b": 3.0},
                                                {"x": (0, 2)})
    assert best["x"] == 2.0
    assert achieved["b"] == 2.0 and residuals["b"] == -1.0
    assert best["distance"] == 1.0

    opt, best, _, _, evaluations = _match(lambda x: (x["x"] - 1) ** 2, {"b": 0.0},
                                           {"x": (-2, 2)}, options={"maxiter": 0, "ftol": 0})
    assert not opt.success
    assert np.isfinite(best["distance"])
    assert len(evaluations) >= 1


def test_invalid_covariance_and_bounds_are_refused():
    with pytest.raises(ValueError, match="shape"):
        _match(lambda x: x["x"], {"b": 1.0}, {"x": (0, 2)}, covariance=np.eye(2))
    with pytest.raises(ValueError, match="semidefinite"):
        _match(lambda x: x["x"], {"b": 1.0}, {"x": (0, 2)}, covariance=[[-1.0]])
    with pytest.raises(ValueError, match="positive"):
        _match(lambda x: x["x"], {"b": 1.0}, {"x": (0, 2)}, covariance=[0.0])
    with pytest.raises(ValueError, match="low < high"):
        _match(lambda x: x["x"], {"b": 1.0}, {"x": (2, 2)})


def test_cge_target_recovers_the_elasticity_and_spec_sweep_result():
    sam, econ = toy.build()
    spec = toy.spec(sigma_armington=2.2)
    scenario = cge.Scenario(world_price=np.array([1.0, 1.2]))

    def measure(base, alt):
        return 100 * (cge.real_gdp(alt)["expenditure"]
                      / cge.real_gdp(base)["expenditure"] - 1)

    calibrated = cge.calibrate(sam, spec)
    base = cge.solve_period(calibrated)
    alt = cge.solve_period(calibrated, scenario=scenario, warm_start=base)
    target = measure(base, alt)
    result = cge.match_targets(sam, toy.spec(), targets={"gdp_pct": target},
                               parameters={"sigma_armington": (1.5, 2.5)},
                               measure=measure, scenario=scenario)
    assert result.success, result.message
    assert result.parameters["sigma_armington"] == pytest.approx(2.2, abs=1e-4)
    assert result.achieved["gdp_pct"] == pytest.approx(target, abs=1e-4)
    assert result.target["gdp_pct"] == target
    assert result.residuals["gdp_pct"] == pytest.approx(0.0, abs=1e-4)
    assert result.distance == pytest.approx(0.0, abs=1e-7)
    assert result.spec.sigma_armington == pytest.approx(2.2, abs=1e-4)
    assert result.nfev == len(result.evaluations)
    assert result.evaluations.distance.min() == pytest.approx(result.distance)


def test_missing_measure_target_fails_at_the_initial_point():
    sam, _ = toy.build()
    with pytest.raises(RuntimeError, match="initial parameter midpoint"):
        cge.match_targets(sam, toy.spec(), targets={"missing": 1.0},
                          parameters={"sigma_va": (0.2, 1.2)}, measure=lambda b, a: {})
