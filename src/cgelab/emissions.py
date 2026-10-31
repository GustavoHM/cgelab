"""Emissions accounts and a carbon price.

A satellite account: emissions per unit of each flow that burns or releases
something, calibrated from base-year totals. Four sources:

* ``intermediate`` -- per unit of commodity ``c`` used by activity ``j``
  (combustion in production), ``(n_com, n_act)``;
* ``household`` and ``government`` -- per unit of commodity bought for final
  use, ``(n_com,)``;
* ``process`` -- per unit of activity output, released whatever the inputs
  (cement, fermentation), ``(n_act,)``.

A carbon price (``Scenario.carbon_price``, numeraire units per unit of
emissions) is a specific tax on each source. On an activity it is a tax per
unit of output equal to the price times the activity's emission intensity:
with Leontief intermediates, taxing the fuel an activity burns and taxing its
output by the fuel's emissions per unit are the same thing, so the carbon price
enters through the activity tax. On households it is a specific tax on the fuel
they buy. Government's own fuel use emits, and is not taxed: it would pay the
tax to itself. The revenue accrues to government; a revenue-neutral closure
(``Closure(government_target="revenue")``) recycles it by cutting another tax.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

__all__ = ["Emissions", "add_emissions", "emissions", "carbon_wedges"]


@dataclass
class Emissions:
    """Emission coefficients, per unit of the base-year quantity of each flow."""

    intermediate: np.ndarray
    household: np.ndarray
    government: np.ndarray
    process: np.ndarray
    unit: str = "tCO2"

    def intensity(self, io_eff: np.ndarray) -> np.ndarray:
        """Emissions per unit of each activity's output, at these input coefficients."""
        return (self.intermediate * io_eff).sum(axis=0) + self.process


def _base_quantities(econ):
    p0 = 1.0 + econ.ptax
    hh = (econ.hh_budget * np.asarray(econ.hh_spend0, float)[:, None]).sum(axis=0) / p0
    return dict(intermediate=econ.io * econ.X0[None, :], household=hh,
                government=econ.gov_dem / p0, process=econ.X0)


def _as_array(value, econ, shape, kind):
    if value is None:
        return np.zeros(shape)
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=econ.coms, columns=econ.acts).fillna(0.0).to_numpy(float)
    if isinstance(value, (pd.Series, dict)):
        names = econ.acts if kind == "process" else econ.coms
        s = pd.Series(value, dtype=float)
        unknown = sorted(set(s.index) - set(names))
        if unknown:
            raise KeyError(f"{kind} emissions for accounts not in the model: {unknown}")
        return s.reindex(names).fillna(0.0).to_numpy(float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{kind} emissions must have shape {shape}, got {arr.shape}")
    return arr


def add_emissions(econ, *, intermediate=None, household=None, government=None, process=None,
                  unit: str = "tCO2"):
    """Return ``econ`` with an emissions account calibrated from base-year totals.

    ``intermediate`` is a DataFrame of commodities by activities (or an array of
    that shape); the others are Series or mappings by account name. Every
    coefficient is the total divided by the base-year quantity of its flow, so
    at the base year the account returns the totals it was given. A nonzero
    total on a flow that is zero in the SAM has no coefficient, and is refused.
    """
    nc, na = econ.n_com, econ.n_act
    totals = dict(intermediate=_as_array(intermediate, econ, (nc, na), "intermediate"),
                  household=_as_array(household, econ, (nc,), "household"),
                  government=_as_array(government, econ, (nc,), "government"),
                  process=_as_array(process, econ, (na,), "process"))
    base = _base_quantities(econ)
    coef = {}
    for k, total in totals.items():
        q = base[k]
        orphan = (total != 0) & (q <= 0)
        if orphan.any():
            raise ValueError(f"{k} emissions on {int(orphan.sum())} flow(s) the SAM records "
                             f"as zero: there is no quantity to attach them to")
        with np.errstate(divide="ignore", invalid="ignore"):
            coef[k] = np.where(q > 0, total / np.where(q > 0, q, 1.0), 0.0)
    # A backstop built like another activity burns its inputs as that activity
    # does, and releases what it releases per unit of output. One built from
    # costs emits nothing: the SAM records no flow to attach emissions to.
    for j, b in getattr(econ, "backstops", {}).items():
        if b["like"] is not None:
            coef["intermediate"][:, j] = coef["intermediate"][:, b["like"]]
            coef["process"][j] = coef["process"][b["like"]]
    return replace(econ, emissions=Emissions(**coef, unit=unit))


def carbon_wedges(period):
    """The carbon tax in force in a solved period: per unit of each activity's
    output, and per unit of each commodity households buy, in numeraire units."""
    econ, sc = period.econ, period.scenario
    if sc.carbon_price is None or econ.emissions is None:
        return np.zeros(econ.n_act), np.zeros(econ.n_com)
    price = float(sc.carbon_price) * period.closure.numeraire
    return price * econ.emissions.intensity(_coefficients(period)), price * econ.emissions.household


def _coefficients(period):
    """The input coefficients the equilibrium used: with an energy nest they moved."""
    if period.input_coefficients is not None:
        return period.input_coefficients
    econ, sc = period.econ, period.scenario
    tfp = np.ones(econ.n_act) if sc.productivity is None else np.asarray(sc.productivity, float)
    return econ.io / tfp[None, :]


def emissions(period) -> pd.Series:
    """Emissions of a solved period by source, and their total.

    Intermediate use is the input coefficients the equilibrium used: they follow
    the scenario's productivity and, under an energy nest, fuel prices.
    """
    econ = period.econ
    em = econ.emissions
    if em is None:
        raise ValueError("this economy has no emissions account; see add_emissions")
    by_activity = (em.intermediate * _coefficients(period)).sum(axis=0) * period.X
    out = pd.Series({
        "intermediate": float(by_activity.sum()),
        "household": float(em.household @ period.hh_quantity.sum(axis=0)),
        "government": float(em.government @ period.gov_quantity),
        "process": float(em.process @ period.X),
    }, name=em.unit)
    out["total"] = float(out.sum())
    return out
