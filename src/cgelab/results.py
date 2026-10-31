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

__all__ = ["factor_demand_by_activity", "real_gdp", "real_value_added", "sectoral_path",
           "national_accounts", "price_indices", "government_account"]


def factor_demand_by_activity(period: PeriodResult) -> pd.DataFrame:
    """Factor use by activity. Rows are factors, columns are activities.

    The row sums equal ``period.factor_demand``, except for a factor under a CET
    transformation (``ModelSpec.factor_mobility``), whose ``factor_demand`` is
    its endowment. Productivity is the scenario's,
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
    own_account = float((econ.hh_direct * _activity_scale(period)[None, :]).sum())
    # Current quantities at base-year prices: the Laspeyres volume of
    # `price_indices`. A regional destination's base price includes its margins.
    base = _base_components(econ)
    expenditure = sum((-1.0 if k == "imports" else 1.0) * float(base[k][0] @ q)
                      for k, (_, q) in _components(period).items())
    # The factor services activities use, each at its base-year price of one.
    # For an imperfectly mobile factor under a CET this is not its endowment,
    # which `factor_demand` reports: moving between activities costs services.
    value_added = float(period.factor_demand_by_activity.sum())
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


def real_value_added(period: PeriodResult) -> pd.Series:
    """Real value added by activity, by double deflation, at base-year prices.

    Output at the base-year activity price of one, less the intermediate inputs
    it used at base-year purchaser prices: how a statistical office measures
    value added in volume. Productivity moves it. The ``value_added`` of
    :func:`real_gdp` does not: it is factor use at base-year factor prices, and
    Hicks-neutral productivity changes the output a factor makes, not the
    factor. Activity taxes stay inside, as in gross value added at basic
    prices, so at the base year the sum is ``value_added + activity_tax``.
    """
    econ = period.econ
    coef = econ.io if period.input_coefficients is None else period.input_coefficients
    unit = 1.0 - (1.0 + econ.ptax) @ coef
    return pd.Series(unit * period.X, index=list(econ.acts), name="real_value_added")


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


def _carbon(period):
    from .emissions import carbon_wedges
    return carbon_wedges(period)


def _components(period) -> dict:
    """Final demand at current prices, as ``{component: (prices, quantities)}``.

    Household consumption is priced per household and commodity at what the
    household pays, wedges included, and own-account purchases at the activity
    price. Imports are at the border. In a regional model each destination
    pays its own composite price, so domestic final demand is priced by
    destination channel rather than at the national average.
    """
    econ = period.econ
    if econ.regional is not None:
        real = period.real
        prices = real["destination_price"]
        hh_prices = prices[econ.regional.household_region]
        return {
            "consumption": (hh_prices.ravel(), real["hh_destination_quantity"].ravel()),
            "government": (prices.ravel(), real["gov_destination_quantity"].ravel()),
            "investment": (prices.ravel(), real["inv_destination_quantity"].ravel()),
            "exports": (period.pe, period.exports),
            "imports": (period.exchange_rate * period.world_import_price, period.imports),
        }
    W, Wg = _wedges(period)
    eff = period.pq[None, :] * (1.0 + W + Wg) + _carbon(period)[1][None, :]
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
    if econ.regional is not None:
        reg = econ.regional
        p0 = reg.base_price
        hh_p0 = p0[reg.household_region]
        hh = reg.hh_budget * np.asarray(econ.hh_spend0, float)[:, None] / hh_p0
        gov = np.zeros_like(p0)
        gov[-1], gov[:reg.n_reg] = reg.gov_dem, reg.gov_dem_regional
        inv = np.zeros_like(p0)
        inv[-1], inv[:reg.n_reg] = reg.inv_dem, reg.inv_dem_regional
        return {
            "consumption": (hh_p0.ravel(), hh.ravel()),
            "government": (p0.ravel(), (gov / p0).ravel()),
            "investment": (p0.ravel(), (inv / p0).ravel()),
            "exports": (np.ones(econ.n_com), econ.exp0),
            "imports": (np.ones(econ.n_com), econ.imp0),
        }
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
    the activity tax, and the scarcity rent of any binding capacity limit
    (``capacity_rent``) and the premium of any binding import quota (``quota_rent``),
    whoever holds the licences. The two agree in every equilibrium, not only at the base
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
    product_taxes = (g["product_tax"] + g["tariff"] + g["output_tax"] + wedges
                     + g["carbon_households"])
    factor_income = float(period.factor_prices @ period.factor_demand)
    production_taxes = g["activity_tax"] + g["carbon_production"]
    # A binding capacity limit's scarcity rent is in the activity's price and is
    # paid to the owners of a factor: value added that no factor's price shows.
    capacity_rent = float(sum(c["rent"] * c["output"]
                              for c in (getattr(period, "capacity", None) or {}).values()))
    # So is an import quota's premium: in the price paid at home, not at the border.
    quota_rent = float(sum(q["rent"] for q in (getattr(period, "quota", None) or {}).values()))
    income = (factor_income + capacity_rent + quota_rent + product_taxes
              + production_taxes)
    return {
        **value,
        "gdp_expenditure": expenditure,
        "factor_income": factor_income,
        "capacity_rent": capacity_rent,
        "quota_rent": quota_rent,
        "taxes_on_products": product_taxes,
        "taxes_on_production": production_taxes,
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
    order = ["product_tax", "activity_tax", "carbon_production", "carbon_households",
             "tariff", "output_tax", "direct_tax",
             "factor_income", "wedge", "quota_rent", "revenue", "transfers", "consumption",
             "absorbed", "saving"]
    # The regional solver has no quotas and books no quota rent.
    return pd.Series({k: period.government.get(k, 0.0) for k in order}, name="government")
