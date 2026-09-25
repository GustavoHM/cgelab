"""Recursive dynamics: a sequence of within-period equilibria linked by capital.

Each year is solved as a static equilibrium; what carries across years is the
capital stock, the labour endowment and total factor productivity. No agent
optimises over the horizon, which is the defining limitation of a
recursive-dynamic CGE and should be stated in any paper that uses one: the model
answers what an economy with these endowments looks like in each year, not what a
forward-looking agent would choose.

Units, and the mistake they invite
----------------------------------
A SAM's capital row is the annual capital SERVICE payment, not the capital stock.
Adding investment to it directly compounds a flow onto a flow and the stock runs
away -- in a twenty-five year horizon by orders of magnitude. The stock is
recovered from the user cost, ``r = discount + depreciation``, and services then
scale with the stock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .calibrate import Economy
from .closure import Closure
from .scenario import Scenario
from .solve import PeriodResult, solve_period

__all__ = ["Growth", "PathResult", "run_path"]


@dataclass
class Growth:
    """Between-period laws of motion for the exogenous supplies."""

    labour: float = 0.0            # annual growth rate of the labour endowment
    tfp: float = 0.0               # annual productivity growth; see `productivity`
    depreciation: float = 0.05
    discount: float = 0.05
    labour_factor: str = "f_lab"
    capital_factor: str = "f_cap"

    #: How productivity growth enters. ``"hicks_neutral"`` scales every input
    #: requirement of every activity, which is technical change proper.
    #: ``"factor_augmenting"`` scales the factor ENDOWMENTS instead, so the
    #: economy grows because it has more effective labour and capital.
    #:
    #: The two are not interchangeable and the difference is visible in prices,
    #: not only in labels. Under Hicks-neutral change the same endowment
    #: produces more, so factor prices rise against the numeraire; under
    #: endowment scaling the price system is homogeneous in the endowment
    #: vector and relative factor prices are governed only by the CHANGING
    #: composition of supply. A model that grows one way and reports the other
    #: is describing a different economy.
    productivity: str = "hicks_neutral"

    #: Deflator that turns nominal saving into the real investment which adds to
    #: the capital stock. ``"base_price"`` values the investment bundle at
    #: base-year purchaser prices, which keeps the stock on the same ruler as
    #: every other real magnitude the solver reports. ``"own_price"`` divides
    #: nominal saving by the CURRENT price of the investment bundle, which is
    #: the volume actually bought that year.
    #:
    #: They coincide in the base year and drift apart with relative prices. The
    #: choice belongs to the modeller: the first makes the capital series
    #: commensurable with real absorption, the second makes it the physical
    #: quantity accumulated.
    investment_deflator: str = "base_price"

    def __post_init__(self):
        for name, allowed in (("productivity", ("hicks_neutral", "factor_augmenting")),
                              ("investment_deflator", ("base_price", "own_price"))):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"Growth.{name} must be one of {allowed}, got {got!r}")

    @property
    def user_cost(self) -> float:
        return self.discount + self.depreciation


@dataclass
class PathResult:
    """A solved path: the per-year equilibria plus a tidy summary."""

    years: list
    periods: list = field(repr=False)
    econ: Economy = field(repr=False)
    growth: Growth = field(repr=False)
    capital_stock: np.ndarray = field(repr=False, default=None)

    def __getitem__(self, year) -> PeriodResult:
        return self.periods[self.years.index(year)]

    @property
    def converged(self) -> bool:
        return all(p.converged for p in self.periods)

    @property
    def worst_walras(self) -> float:
        return max(abs(p.walras) for p in self.periods)

    def aggregates(self) -> pd.DataFrame:
        """One row per year: prices, real absorption, savings and the Walras residual."""
        rows = []
        for year, p in zip(self.years, self.periods):
            row = {"year": year, "exchange_rate": p.exchange_rate,
                   "savings": p.savings, "walras": p.walras,
                   "residual": p.residual, "converged": p.converged}
            for f, v in zip(self.econ.factors, p.factor_prices):
                row[f"price_{f}"] = v
            row.update({k: v for k, v in p.real.items() if np.isscalar(v)})
            rows.append(row)
        return pd.DataFrame(rows).set_index("year")

    def households(self) -> pd.DataFrame:
        """One row per year and household account: the inputs a welfare measure needs."""
        rows = []
        for year, p in zip(self.years, self.periods):
            for i, h in enumerate(self.econ.hhs):
                rows.append({
                    "year": year, "account": h,
                    "income": float(p.hh_income[i]),
                    "direct_outlay": float(p.hh_direct_outlay[i]),
                    "commodity_spend": float(p.hh_commodity_spend[i]),
                    "price_index": float(p.real["hh_price_index"][i]),
                    "wedge_paid": float(p.hh_wedge_paid[i]),
                })
        return pd.DataFrame(rows)

    def commodity(self, name: str) -> pd.DataFrame:
        """Price and output of one commodity along the path."""
        j = self.econ.commodity(name)
        return pd.DataFrame({"year": self.years,
                             "price": [float(p.pq[j]) for p in self.periods],
                             "output": [float(p.Q[j]) for p in self.periods]}
                            ).set_index("year")


def run_path(econ: Economy, closure: Closure | None = None, *, years: Sequence,
             scenario: Scenario | Callable | None = None,
             growth: Growth | None = None, raise_on_fail: bool = True) -> PathResult:
    """Solve a sequence of years, accumulating capital between them.

    Parameters
    ----------
    scenario : Scenario or callable
        A single scenario applied every year, or ``f(year, index) -> Scenario``.
        The callable form is what a policy with a statutory CALENDAR needs.
    growth : Growth
        Laws of motion for labour, productivity and the capital stock.

    Notes
    -----
    Factor supplies set inside a per-year scenario OVERRIDE the accumulated ones,
    which is deliberate: an exercise that fixes an endowment by hand should not
    have the growth block quietly undo it. Anything the scenario leaves as
    ``None`` is filled from the path.
    """
    closure = closure or Closure()
    growth = growth or Growth()
    years = list(years)
    ki = econ.factor(growth.capital_factor)
    li = econ.factor(growth.labour_factor)

    services0 = float(econ.fac_supply[ki])
    stock = services0 / growth.user_cost
    labour0 = float(econ.fac_supply[li])

    periods, stocks = [], []
    warm = None
    for t, year in enumerate(years):
        sc = scenario(year, t) if callable(scenario) else (scenario or Scenario())
        level = (1.0 + growth.tfp) ** t
        supply = econ.fac_supply.copy()
        supply[li] = labour0 * (1.0 + growth.labour) ** t
        supply[ki] = stock * growth.user_cost
        if growth.productivity == "factor_augmenting":
            supply = supply * level
            tfp = np.ones(econ.n_act)
        else:
            tfp = np.full(econ.n_act, level)
        if sc.factor_supply is not None:
            supply = np.asarray(sc.factor_supply, dtype=float)
        if sc.productivity is not None:
            tfp = tfp * np.asarray(sc.productivity, dtype=float)
        sc = sc.with_(factor_supply=supply, productivity=tfp)

        res = solve_period(econ, closure, sc, warm_start=warm, raise_on_fail=raise_on_fail)
        periods.append(res)
        stocks.append(stock)
        warm = res

        # Real investment adds to the stock. The flow-onto-flow error this avoids
        # -- adding nominal saving to the capital SERVICE payment -- is the single
        # most common bug in a recursive dynamic.
        if growth.investment_deflator == "own_price":
            inv_w = econ.inv_dem / max(float(econ.inv_dem.sum()), 1e-12)
            p_inv = float(inv_w @ res.pqf)
            inv_real = res.savings / max(p_inv, 1e-12)
        else:
            inv_real = float(res.real["inv_real"])
        stock = (1.0 - growth.depreciation) * stock + inv_real

    return PathResult(years=years, periods=periods, econ=econ, growth=growth,
                      capital_stock=np.array(stocks))
