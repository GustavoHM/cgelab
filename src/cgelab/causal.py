"""Treatment-effect contrasts from solved regional equilibria."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["treated_vs_control", "simulated_panel"]


def _region_outcome(period, region: str, measure):
    """One regional outcome, shared with the simulated-panel extractor."""
    regional = period.econ.regional
    i = regional.regions.index(region)
    if callable(measure):
        return float(measure(period, region))
    if measure == "output":
        return float(period.X[regional.activity_region == i].sum())
    if measure == "real_income":
        return float(np.asarray(period.real["hh_real_income"])[regional.household_region == i].sum())
    if measure == "factor_demand":
        where = [j for j, name in enumerate(period.econ.factors)
                 if name.partition(":")[0] == region]
        return float(period.factor_demand[where].sum())
    if measure == "bilateral":
        return float(period.bilateral[i, :, :].sum())
    raise ValueError(f"unknown regional measure {measure!r}; use a callable or "
                     "'output', 'real_income', 'factor_demand' or 'bilateral'")


def _regions(reference, alternative, treated, controls):
    if reference.econ.regional is None or alternative.econ.regional is None:
        raise ValueError("treated_vs_control needs two solved regional periods")
    if (reference.econ.acts != alternative.econ.acts
            or reference.econ.coms != alternative.econ.coms
            or reference.econ.hhs != alternative.econ.hhs
            or reference.econ.factors != alternative.econ.factors):
        raise ValueError("reference and alternative have different accounts")
    regions = list(reference.econ.regional.regions)
    if regions != list(alternative.econ.regional.regions):
        raise ValueError("reference and alternative have different regions")
    controls = [controls] if isinstance(controls, str) else list(controls)
    if not controls:
        raise ValueError("controls must contain at least one region")
    if treated in controls:
        raise ValueError("the treated region must not be one of the controls")
    missing = [region for region in [treated] + controls if region not in regions]
    if missing:
        raise KeyError(f"regions not in the model: {missing}")
    return controls


def _control_weights(controls, weights):
    if weights is None:
        return {region: 1.0 / len(controls) for region in controls}
    if not isinstance(weights, Mapping) or set(weights) != set(controls):
        raise ValueError("weights must name every control exactly once")
    values = {region: float(weights[region]) for region in controls}
    if not np.isfinite(list(values.values())).all() or any(v < 0 for v in values.values()):
        raise ValueError("control weights must be finite and nonnegative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("control weights must have a positive sum")
    return {region: value / total for region, value in values.items()}


def treated_vs_control(reference, alternative, *, treated: str, controls,
                       measure="output", weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """The treated change minus the weighted control change between two periods.

    ``measure`` is a built-in outcome name, a callable ``f(period, region)``, or
    a mapping of output names to either. Built-ins are ``output`` (activity
    output), ``real_income`` (household real income), ``factor_demand`` and
    ``bilateral`` (outflows from the region). Controls are equally weighted by
    default; explicit weights are normalised to sum to one. The function reports
    a contrast of solved outcomes, not an econometric identification strategy.
    """
    controls = _regions(reference, alternative, treated, controls)
    weights = _control_weights(controls, weights)
    if isinstance(measure, Mapping):
        measures = dict(measure)
    elif callable(measure) or isinstance(measure, str):
        measures = {measure if isinstance(measure, str) else "effect": measure}
    else:
        raise TypeError("measure must be a name, callable or mapping")
    if not measures:
        raise ValueError("measure must contain at least one outcome")
    rows = {}
    for name, definition in measures.items():
        ref_t = _region_outcome(reference, treated, definition)
        alt_t = _region_outcome(alternative, treated, definition)
        ref_c = sum(weights[r] * _region_outcome(reference, r, definition) for r in controls)
        alt_c = sum(weights[r] * _region_outcome(alternative, r, definition) for r in controls)
        rows[name] = {"treated_change": alt_t - ref_t, "control_change": alt_c - ref_c,
                      "effect": (alt_t - ref_t) - (alt_c - ref_c),
                      "reference_treated": ref_t, "alternative_treated": alt_t,
                      "reference_control": ref_c, "alternative_control": alt_c}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("outcome")


def simulated_panel(path, *, measure="output", treated=None,
                    noise: float = 0.0, seed: int | None = 20260724) -> pd.DataFrame:
    """A tidy region-by-year panel from a regional ``PathResult``.

    ``measure`` has the same built-ins and callable forms as
    :func:`treated_vs_control`. ``treated`` only labels rows; it never changes
    the model. ``noise`` is the log standard deviation of independent lognormal
    multipliers with mean one, applied only to reported values after every
    equilibrium and capital stock has been solved.
    """
    if not hasattr(path, "years") or not hasattr(path, "periods"):
        raise TypeError("simulated_panel expects a PathResult")
    if path.econ.regional is None:
        raise ValueError("simulated_panel needs a regional path")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if isinstance(measure, Mapping):
        measures = dict(measure)
    elif callable(measure) or isinstance(measure, str):
        measures = {measure if isinstance(measure, str) else "effect": measure}
    else:
        raise TypeError("measure must be a name, callable or mapping")
    if not measures:
        raise ValueError("measure must contain at least one outcome")
    regions = list(path.econ.regional.regions)
    if treated is not None and treated not in regions:
        raise KeyError(f"treated region {treated!r} is not in the model")
    rng = np.random.default_rng(seed)
    rows = []
    for year, period in zip(path.years, path.periods):
        for region in regions:
            for name, definition in measures.items():
                value = _region_outcome(period, region, definition)
                shock = (float(rng.lognormal(-0.5 * noise ** 2, noise)) if noise else 1.0)
                rows.append({"year": year, "region": region, "outcome": name,
                             "value": value * shock, "treated": region == treated,
                             "noise": shock})
    return pd.DataFrame(rows, columns=["year", "region", "outcome", "value", "treated", "noise"])
