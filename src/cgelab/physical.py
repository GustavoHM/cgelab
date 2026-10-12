"""Physical units: kilowatt-hours, tonnes, kilograms, alongside the money.

A satellite account like the emissions one, calibrated from base-year
physical totals. Two anchors, because the model's money units differ by flow
(producer, border and tax-inclusive values):

* for a COMMODITY, physical content per unit of its Armington composite -- the
  domestic and imported variety that activities and households buy -- from its
  physical supply (domestic sales plus imports) at the base year;
* for an ACTIVITY, physical output per unit of its activity level, from its
  physical output at the base year.

The link is fixed: a unit of composite carries the same physical content in
every period. That is right for a homogeneous good (a kilowatt-hour is a
kilowatt-hour) and wrong for a composite of qualities, which is why the account
is opt-in, commodity by commodity.

:func:`physical_coefficient` runs the other way, for building a SAM: it turns
a technical coefficient stated in physical units -- kilowatt-hours of electricity
per kilogram of output -- into the money coefficient of an intermediate entry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["PhysicalUnits", "add_physical_units", "physical", "physical_coefficient"]


@dataclass
class PhysicalUnits:
    """Physical content per model unit, by commodity composite and by activity output."""

    commodities: dict          # name -> (unit, per unit of composite)
    activities: dict           # name -> (unit, per unit of activity level)


def add_physical_units(econ, *, commodities: Mapping[str, tuple] | None = None,
                       activities: Mapping[str, tuple] | None = None):
    """Return ``econ`` with physical units calibrated from base-year totals.

    ``commodities`` maps a commodity to ``(unit, physical supply)``: its
    domestic sales plus imports in the base year, in that unit. ``activities``
    maps an activity to ``(unit, physical output)``.
    """
    com_out, act_out = {}, {}
    composite0 = econ.arm_share * econ.A0
    for name, (unit, total) in (commodities or {}).items():
        c = econ.commodity(name)
        if composite0[c] <= 0:
            raise ValueError(f"{name} has no domestic sales or imports to attach a unit to")
        com_out[name] = (unit, float(total) / float(composite0[c]))
    for name, (unit, total) in (activities or {}).items():
        j = econ.activity(name)
        if econ.X0[j] <= 0:
            raise ValueError(f"{name} produces nothing in the base year")
        act_out[name] = (unit, float(total) / float(econ.X0[j]))
    return replace(econ, physical_units=PhysicalUnits(com_out, act_out))


def physical(period) -> dict:
    """Physical quantities of a solved period.

    ``supply`` and ``use`` by commodity (use split by activity, households and
    government, investment), ``output`` by activity, and ``intensity``: a
    commodity's physical use per physical unit of each activity's output, where
    both have units.
    """
    econ = period.econ
    pu = econ.physical_units
    if pu is None:
        raise ValueError("this economy has no physical units; see add_physical_units")
    coef = (period.input_coefficients if period.input_coefficients is not None
            else econ.io)
    supply, use, output = {}, {}, {}
    for name, (unit, per_composite) in pu.commodities.items():
        c = econ.commodity(name)
        supply[name] = (unit, float(econ.arm_share[c] * period.absorption[c] * per_composite))
        # Uses are in absorption units, and a unit of absorption holds arm_share
        # units of the composite; the rest is the margins delivered with it.
        per = per_composite * float(econ.arm_share[c])
        by_act = coef[c] * period.X * per
        use[name] = pd.Series(
            {**dict(zip(econ.acts, by_act)),
             "households": float(period.hh_quantity[:, c].sum() * per),
             "government": float(period.gov_quantity[c] * per),
             "investment": float(period.investment_quantity[c] * per)}, name=unit)
    for name, (unit, per) in pu.activities.items():
        output[name] = (unit, float(period.X[econ.activity(name)] * per))
    intensity = {}
    for cname, (cunit, cper) in pu.commodities.items():
        c = econ.commodity(cname)
        cper = cper * float(econ.arm_share[c])
        for aname, (aunit, aper) in pu.activities.items():
            j = econ.activity(aname)
            intensity[(cname, aname)] = (f"{cunit}/{aunit}", float(coef[c, j] * cper / aper))
    return dict(supply=supply, use=use, output=output, intensity=intensity)


def physical_coefficient(amount: float, *, input_per_value: float, output_per_value: float,
                         input_price: float = 1.0) -> float:
    """Money input per unit of output value, from a physical technical coefficient.

    ``amount`` is physical input per physical unit of output (kWh per kg, say).
    ``input_per_value`` is the input's physical content per unit of its
    absorption at base prices (kWh per currency unit of electricity, before
    product tax), ``output_per_value`` the output's per unit of activity output
    (kg per currency unit of the good). ``input_price`` is the purchaser price
    of a unit of absorption at the base year (``1 + ptax``), so the result is the
    SAM entry per unit of output value, at purchaser prices.
    """
    if input_per_value <= 0 or output_per_value <= 0:
        raise ValueError("physical content per unit of value must be positive")
    return float(amount) * output_per_value / input_per_value * input_price
