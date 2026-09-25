"""What a counterfactual changes, relative to the calibrated base year.

Everything a policy experiment can move lives in one object. That is worth the
small ceremony: a scenario written as keyword arguments scattered over a solver
call cannot be logged, compared, or handed to a robustness sweep, and the
difference between two runs then exists only in the code that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

__all__ = ["Scenario", "consumer_wedge"]


@dataclass
class Scenario:
    """A counterfactual. Every field defaults to "no change"."""

    name: str = "baseline"

    #: Factor supplies, absolute levels. Absent, the calibrated supplies hold.
    factor_supply: np.ndarray | None = None

    #: Ownership of each factor, ``(n_hh, n_fac)``, columns summing to one.
    #: The hook for an extensive-margin story in which households MOVE between
    #: accounts -- adopters and non-adopters, formal and informal -- rather than
    #: each existing household doing more of something.
    factor_ownership: np.ndarray | None = None

    #: Hicks-neutral productivity multiplier per activity.
    productivity: np.ndarray | None = None

    #: Activity levels set exogenously, as a multiple of the base year. Only for
    #: activities that sell DIRECTLY to households and have no market share in
    #: any commodity -- own-account production. The solver refuses otherwise,
    #: because scaling an activity that supplies the commodity market while the
    #: commodity balance still determines its level is a contradiction, not an
    #: approximation.
    activity_scale: Mapping[str, float] = field(default_factory=dict)

    #: Product tax rates facing FINAL consumers. Absent, the calibrated rates.
    product_tax: np.ndarray | None = None

    #: The NON-CREDITABLE part of the product tax, that is, what actually
    #: burdens a buyer purchasing for production. Under a cascading turnover tax
    #: the two coincide and the two-price block collapses to the single-price
    #: one exactly; under a fully non-cumulative VAT only the residual old
    #: system cascades. Absent, equals ``product_tax``.
    product_tax_firm: np.ndarray | None = None

    #: Ad valorem wedge on household purchases, ``(n_hh, n_com)``, whose revenue
    #: is recycled to households by ``wedge_revenue_to``.
    consumer_wedge: np.ndarray | None = None

    #: Second wedge whose revenue accrues to GOVERNMENT rather than to
    #: households -- a refund granted at the till, a levy earmarked to the
    #: budget. Kept apart from ``consumer_wedge`` because routing a refund
    #: through household recycling would credit the money to the wrong agent.
    government_wedge: np.ndarray | None = None

    #: Who receives the recycled wedge revenue: ``"capital"`` (in proportion to
    #: ownership of the named factor), ``"lumpsum"`` (per capita), a float in
    #: [0, 1] mixing the two, or a callable returning shares over households.
    #:
    #: Not a detail of implementation. Capital ownership is concentrated at the
    #: top, so under ``"capital"`` the poor pay a wedge on their purchases while
    #: the rich receive it as income, and part of any measured regressivity is
    #: manufactured by that assumption. A robustness exercise that does not vary
    #: this is not a robustness exercise.
    wedge_revenue_to: str | float | Callable = "capital"

    #: Factor whose ownership defines the ``"capital"`` recycling rule.
    wedge_revenue_factor: str | None = None

    #: Lump-sum transfer to each household, positive for a receipt. The general
    #: hook for a named agent bearing a shortfall.
    transfers: np.ndarray | None = None

    #: Amount charged to government (lowering public saving), positive for a cost.
    government_absorbs: float = 0.0

    #: World price of each imported commodity, as a multiple of the base year.
    world_price: np.ndarray | None = None

    #: World price of each EXPORTED commodity, as a multiple of the base year.
    #: Read only under ``export_rule="cet"``: with fixed export shares exports
    #: are a fixed fraction of output whatever they fetch abroad, and the field
    #: is refused rather than silently ignored.
    export_price: np.ndarray | None = None

    #: (n_fac,) share of each factor's income owned by the rest of the world,
    #: overriding the SAM's. `run_path` sets it every year from the foreign-owned
    #: capital stock it tracks, unless the scenario sets it explicitly.
    foreign_factor_share: np.ndarray | None = None

    #: Foreign direct investment this period, ``{capital factor: amount}`` in
    #: numeraire units. It raises foreign saving by the amount -- so it finances
    #: investment -- and `run_path` adds the capital it buys to the FOREIGN-owned
    #: stock of that factor, whose returns then leave the economy. FDI that only
    #: raised foreign saving would hand residents a capital stock paid for from
    #: abroad, and every gain it produced would be counted as theirs.
    fdi: Mapping[str, float] = field(default_factory=dict)

    def with_(self, **changes) -> "Scenario":
        """Return a copy with fields replaced -- for building a sweep from a base case."""
        import dataclasses
        return dataclasses.replace(self, **changes)


def consumer_wedge(econ, rates: Mapping[str, float], households=None) -> np.ndarray:
    """Build an ``(n_hh, n_com)`` wedge from commodity -> ad valorem rate.

    ``households`` restricts the wedge to a subset of accounts; absent, it
    applies to all of them.
    """
    W = np.zeros((econ.n_hh, econ.n_com))
    rows = (range(econ.n_hh) if households is None
            else [econ.household(h) if isinstance(h, str) else h for h in households])
    for com, rate in rates.items():
        j = econ.commodity(com) if isinstance(com, str) else com
        for i in rows:
            W[i, j] = float(rate)
    return W
