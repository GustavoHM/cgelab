"""Results an applied model otherwise rebuilds by hand.

Three numbers kept drifting apart between users of the same solved path: employment
by sector, real GDP, and the sectoral price and output panel. Each is a function
of a :class:`~cgelab.solve.PeriodResult` or a :class:`~cgelab.dynamics.PathResult`,
not a second pass of the solver, so two people reading one equilibrium get the
same figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .solve import PeriodResult

__all__ = ["factor_demand_by_activity", "real_gdp", "sectoral_path"]


def factor_demand_by_activity(period: PeriodResult) -> pd.DataFrame:
    """Factor use by activity. Rows are factors, columns are activities.

    The row sums equal ``period.factor_demand``. Productivity is the scenario's,
    because the matrix stored on the result was built with the same coefficients
    the equilibrium was solved at.
    """
    econ = period.econ
    return pd.DataFrame(period.factor_demand_by_activity,
                        index=list(econ.factors), columns=list(econ.acts))


def real_gdp(period: PeriodResult) -> dict:
    """Real GDP from the expenditure and the value-added sides, and what separates them.

    Expenditure is at base-year prices: household, government and investment demand
    at base-year purchaser prices (``1 + ptax``), exports and imports at the
    base-year basic price of 1, plus own-account purchases valued at the base-year
    activity price of 1. Leaving own-account out makes the value-added side larger
    by exactly that activity and the gap looks like a leak.

    Value added is primary-factor use valued at base-year factor prices, which are
    1 in the model's units. It is not the value-added bundle ``vunit @ X``: that
    bundle ignores substitution, and this is the factor bill the SAM would record
    if base-year factor prices still held.

    The gap is expenditure minus value added. At the base year it equals product
    tax plus activity tax, both at the calibrated rates on the solved quantities,
    and nothing else: margins sit on both sides (inside purchaser prices and inside
    the margin activity's value added) and trade is valued at base prices, so
    neither margins nor the terms of trade enter. Away from the base year a small
    ``other`` remains, the base-price identity's residual once quantities leave
    the calibrated point. It is reported rather than folded into the tax.
    """
    econ = period.econ
    yard = 1.0 + econ.ptax
    final = (period.hh_quantity.sum(axis=0) + period.gov_quantity
             + period.investment_quantity)
    scale = np.ones(econ.n_act)
    for name, mult in period.scenario.activity_scale.items():
        j = econ.activity(name) if isinstance(name, str) else int(name)
        scale[j] = float(mult)
    own_account = float((econ.hh_direct * scale[None, :]).sum())
    expenditure = float(yard @ final + period.exports.sum() - period.imports.sum()
                        + own_account)
    value_added = float(period.factor_demand.sum())
    product_tax = float(econ.ptax @ period.absorption)
    activity_tax = float(econ.atax @ period.X)
    gap = expenditure - value_added
    other = gap - product_tax - activity_tax
    return {
        "expenditure": expenditure,
        "value_added": value_added,
        "gap": gap,
        "product_tax": product_tax,
        "activity_tax": activity_tax,
        "own_account": own_account,
        "other": other,
    }


def sectoral_path(path) -> pd.DataFrame:
    """One row per year and account: output and prices along a solved path.

    Commodities carry ``Q``, ``pd`` and ``pq``. Activities carry ``X`` and ``pa``.
    The column that does not apply to a row is left missing, so the frame stays
    one table instead of two that a user then has to join by year.
    """
    econ = path.econ
    rows = []
    for year, period in zip(path.years, path.periods):
        for j, name in enumerate(econ.coms):
            rows.append({
                "year": year, "kind": "commodity", "account": name,
                "Q": float(period.Q[j]), "X": np.nan,
                "pd": float(period.pd[j]), "pq": float(period.pq[j]),
                "pa": np.nan,
            })
        for j, name in enumerate(econ.acts):
            rows.append({
                "year": year, "kind": "activity", "account": name,
                "Q": np.nan, "X": float(period.X[j]),
                "pd": np.nan, "pq": np.nan, "pa": float(period.pa[j]),
            })
    return pd.DataFrame(rows)
