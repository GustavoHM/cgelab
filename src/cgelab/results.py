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

__all__ = ["factor_demand_by_activity", "real_gdp", "sectoral_path", "national_accounts",
           "price_indices", "government_account"]


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


# ---------------------------------------------------------------------------
# National accounts, price indices and the government account
# ---------------------------------------------------------------------------

def _activity_scale(period) -> np.ndarray:
    econ = period.econ
    scale = np.ones(econ.n_act)
    for name, mult in period.scenario.activity_scale.items():
        j = econ.activity(name) if isinstance(name, str) else int(name)
        scale[j] = float(mult)
    return scale


def _wedges(period):
    econ, sc = period.econ, period.scenario
    W = np.zeros((econ.n_hh, econ.n_com)) if sc.consumer_wedge is None else np.asarray(
        sc.consumer_wedge, float)
    Wg = np.zeros_like(W) if sc.government_wedge is None else np.asarray(
        sc.government_wedge, float)
    return W, Wg


def _components(period) -> dict:
    """Final demand at current prices, as ``{component: (prices, quantities)}``.

    Household consumption is priced per household and commodity at what the
    household pays, wedges included, and own-account purchases at the activity
    price. Imports are at the border.
    """
    econ = period.econ
    W, Wg = _wedges(period)
    eff = period.pq[None, :] * (1.0 + W + Wg)
    own_q = econ.hh_direct.sum(axis=0) * _activity_scale(period)
    keep = own_q != 0
    return {
        "consumption": (np.concatenate([eff.ravel(), period.pa[keep]]),
                        np.concatenate([period.hh_quantity.ravel(), own_q[keep]])),
        "government": (period.pq, period.gov_quantity),
        "investment": (period.pqf, period.investment_quantity),
        "exports": (period.pe, period.exports),
        "imports": (period.exchange_rate * period.world_import_price, period.imports),
    }


def _base_components(econ) -> dict:
    """The same components at the calibrated base year: what the SAM records."""
    p0 = 1.0 + econ.ptax
    hh = econ.hh_budget * np.asarray(econ.hh_spend0, float)[:, None] / p0[None, :]
    own_q = econ.hh_direct.sum(axis=0)
    keep = own_q != 0
    return {
        "consumption": (np.concatenate([np.broadcast_to(p0, hh.shape).ravel(),
                                        np.ones(int(keep.sum()))]),
                        np.concatenate([hh.ravel(), own_q[keep]])),
        "government": (p0, econ.gov_dem / p0),
        "investment": (p0, econ.inv_dem / p0),
        "exports": (np.ones(econ.n_com), econ.exp0),
        "imports": (np.ones(econ.n_com), econ.imp0),
    }


def national_accounts(period: PeriodResult) -> dict:
    """GDP at current market prices, from the expenditure and the income side.

    Expenditure: household consumption at what households pay (consumer wedges
    and own-account purchases included), government consumption, investment,
    exports, less imports at the border. Income: factor income, taxes on
    products (the product tax, tariff, output tax and both consumer wedges) and
    the activity tax. The two agree in every equilibrium, not only at the base
    year; ``gap`` reports what separates them, and a gap above rounding is an
    accounting error. ``gni`` deducts factor income paid abroad.
    """
    comp = _components(period)
    value = {k: float(p @ q) for k, (p, q) in comp.items()}
    expenditure = (value["consumption"] + value["government"] + value["investment"]
                   + value["exports"] - value["imports"])
    W, Wg = _wedges(period)
    g = period.government
    wedges = float((period.hh_quantity * (W + Wg) * period.pq[None, :]).sum())
    product_taxes = g["product_tax"] + g["tariff"] + g["output_tax"] + wedges
    factor_income = float(period.factor_prices @ period.factor_demand)
    income = factor_income + product_taxes + g["activity_tax"]
    return {
        **value,
        "gdp_expenditure": expenditure,
        "factor_income": factor_income,
        "taxes_on_products": product_taxes,
        "taxes_on_production": g["activity_tax"],
        "gdp_income": income,
        "gap": expenditure - income,
        "factor_income_abroad": float(period.factor_income_abroad),
        "gni": income - float(period.factor_income_abroad),
    }


def price_indices(period: PeriodResult, reference: PeriodResult | None = None) -> pd.DataFrame:
    """Laspeyres, Paasche and Fisher price and quantity indices against a reference.

    One row per component of final demand, one for GDP (imports entering with a
    minus sign) and one for the terms of trade (the export index over the import
    index, formula by formula). The reference is another solved period, a
    baseline at the same date for instance; absent, the calibrated base year.

    The Fisher indices pass the factor-reversal test: price times quantity is
    the value ratio. Laspeyres price times Paasche quantity is too, and so is
    the reverse.
    """
    cur = _components(period)
    ref = _base_components(period.econ) if reference is None else _components(reference)
    sign = {"imports": -1.0}
    rows = {}

    def index(p1, q1, p0, q0):
        l_p = (p1 @ q0) / (p0 @ q0)
        p_p = (p1 @ q1) / (p0 @ q1)
        l_q = (p0 @ q1) / (p0 @ q0)
        p_q = (p1 @ q1) / (p1 @ q0)
        return {"laspeyres": l_p, "paasche": p_p, "fisher": float(np.sqrt(l_p * p_p)),
                "quantity_laspeyres": l_q, "quantity_paasche": p_q,
                "quantity_fisher": float(np.sqrt(l_q * p_q)),
                "value": (p1 @ q1) / (p0 @ q0)}

    for k in cur:
        (p1, q1), (p0, q0) = cur[k], ref[k]
        rows[k] = index(p1, q1, p0, q0)
    stack = lambda d, i: np.concatenate([d[k][i] for k in d])  # noqa: E731
    signs = np.concatenate([np.full(cur[k][1].size, sign.get(k, 1.0)) for k in cur])
    rows["gdp"] = index(stack(cur, 0), signs * stack(cur, 1), stack(ref, 0),
                        signs * stack(ref, 1))
    out = pd.DataFrame(rows).T
    tot = out.loc["exports"] / out.loc["imports"]
    tot[["value"]] = np.nan
    out.loc["terms_of_trade"] = tot
    return out


def government_account(period: PeriodResult) -> pd.Series:
    """Revenue by instrument, outlays and saving, at current prices.

    The solver records it at the solution with the formulas the closure solved
    with, so ``saving`` is the government saving the equilibrium financed
    investment with.
    """
    order = ["product_tax", "activity_tax", "tariff", "output_tax", "direct_tax",
             "factor_income", "wedge", "revenue", "transfers", "consumption",
             "absorbed", "saving"]
    return pd.Series({k: period.government[k] for k in order}, name="government")
