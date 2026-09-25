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
from typing import Callable, Mapping, Sequence

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
    #: A rate, or a mapping from capital factor to rate when there are several.
    depreciation: float | Mapping[str, float] = 0.05
    discount: float = 0.05
    labour_factor: str = "f_lab"
    #: One capital factor, or several -- sector-specific capital, a vintage, a
    #: public stock. Each is accumulated separately from its own base-year service
    #: flow, and domestic investment is shared among them by `allocation`.
    capital_factor: str | Sequence[str] = "f_cap"

    #: How domestic investment is shared among several capital factors.
    #: ``"proportional"`` keeps each factor's share of the aggregate stock, so
    #: with no shock the composition of capital never moves. A number ``e`` makes
    #: the share of factor k proportional to ``K_k * (r_k / r_mean) ** e``, with
    #: ``r`` the factor's rental price: investment flows toward capital that earns
    #: more, as in the recursive models of Dervis, de Melo and Robinson. It is the
    #: behavioural parameter of the allocation, and it has no input-output source.
    allocation: str | float = "proportional"

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
        if isinstance(self.allocation, str) and self.allocation != "proportional":
            raise ValueError("Growth.allocation must be 'proportional' or a number, "
                             f"got {self.allocation!r}")
        for name, allowed in (("productivity", ("hicks_neutral", "factor_augmenting")),
                              ("investment_deflator", ("base_price", "own_price"))):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"Growth.{name} must be one of {allowed}, got {got!r}")

    @property
    def capital_factors(self) -> list:
        cf = self.capital_factor
        return [cf] if isinstance(cf, str) else list(cf)

    def depreciation_of(self, factor: str) -> float:
        d = self.depreciation
        if isinstance(d, Mapping):
            if factor not in d:
                raise KeyError(f"no depreciation rate given for capital factor {factor!r}")
            return float(d[factor])
        return float(d)

    @property
    def user_cost(self) -> float:
        """Discount plus depreciation, for the single-capital case."""
        return self.discount + self.depreciation_of(self.capital_factors[0])


@dataclass
class PathResult:
    """A solved path: the per-year equilibria plus a tidy summary."""

    years: list
    periods: list = field(repr=False)
    econ: Economy = field(repr=False)
    growth: Growth = field(repr=False)
    capital_stock: np.ndarray = field(repr=False, default=None)
    #: Foreign-owned part of each capital stock, same shape as `capital_stock`.
    foreign_capital: np.ndarray = field(repr=False, default=None)
    capital_factors: list = field(repr=False, default=None)

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
                    "direct_tax": (0.0 if p.hh_direct_tax is None
                                   else float(p.hh_direct_tax[i])),
                })
        return pd.DataFrame(rows)

    def sectoral(self) -> pd.DataFrame:
        """Output and prices of every account along the path. See :func:`cgekit.results.sectoral_path`."""
        from .results import sectoral_path
        return sectoral_path(self)

    def commodity(self, name: str) -> pd.DataFrame:
        """Price and output of one commodity along the path."""
        j = self.econ.commodity(name)
        return pd.DataFrame({"year": self.years,
                             "price": [float(p.pq[j]) for p in self.periods],
                             "output": [float(p.Q[j]) for p in self.periods]}
                            ).set_index("year")


def run_path(econ: Economy, closure: Closure | None = None, *, years: Sequence,
             scenario: Scenario | Callable | None = None,
             growth: Growth | None = None,
             economy: Callable | None = None,
             raise_on_fail: bool = True) -> PathResult:
    """Solve a sequence of years, accumulating capital between them.

    Parameters
    ----------
    scenario : Scenario or callable
        A single scenario applied every year, or ``f(year, index) -> Scenario``.
        The callable form is what a policy with a statutory CALENDAR needs.
    growth : Growth
        Laws of motion for labour, productivity and the capital stock(s).
    economy : callable, optional
        ``f(year, index) -> Economy``: the economy to solve that year. It is how
        a change that is not a price, tax or endowment enters -- a technology
        adopted over time (input coefficients), a foreign-saving path, a new
        transfer rule. The economy returned must have the same accounts as
        ``econ``; build it with ``dataclasses.replace(econ, ...)``. The base
        ``econ`` fixes the initial capital stocks and labour endowment.

    Notes
    -----
    Factor supplies set inside a per-year scenario OVERRIDE the accumulated ones,
    which is deliberate: an exercise that fixes an endowment by hand should not
    have the growth block quietly undo it. Anything the scenario leaves as
    ``None`` is filled from the path.

    Foreign ownership is tracked per capital factor: the base-year foreign share
    comes from the SAM (``row <- factor`` cells), FDI adds to the foreign-owned
    stock, both depreciate, and each year's ``foreign_factor_share`` is the
    foreign part of the stock -- so the returns to capital that foreigners paid
    for leave the economy through the current account.
    """
    closure = closure or Closure()
    growth = growth or Growth()
    if econ.regional is not None and len(econ.regional.regions) > 1:
        raise ValueError(
            "run_path does not move factors between regions. The within-period "
            "nest is solved by solve_period; a regional capital stock that "
            "accumulates, and migration, are not in this slice.")
    years = list(years)
    caps = growth.capital_factors
    ki = np.array([econ.factor(k) for k in caps])
    li = econ.factor(growth.labour_factor)
    dep = np.array([growth.depreciation_of(k) for k in caps])
    user_cost = growth.discount + dep

    stock = econ.fac_supply[ki] / user_cost
    row_share0 = econ.zeros_if_none("fac_row_share", econ.n_fac)
    foreign = stock * row_share0[ki]
    labour0 = float(econ.fac_supply[li])

    periods, stocks, foreigns = [], [], []
    warm = None
    for t, year in enumerate(years):
        e_t = economy(year, t) if economy is not None else econ
        if e_t.factors != econ.factors or e_t.coms != econ.coms or e_t.acts != econ.acts:
            raise ValueError(f"economy({year!r}, {t}) returned an Economy with different "
                             f"accounts; build it from the base with dataclasses.replace")
        sc = scenario(year, t) if callable(scenario) else (scenario or Scenario())
        unknown = [k for k in sc.fdi if k not in caps]
        if unknown:
            raise ValueError(f"FDI into {unknown}, which Growth does not accumulate as capital "
                             f"(capital factors: {caps})")
        level = (1.0 + growth.tfp) ** t
        supply = e_t.fac_supply.copy()
        supply[li] = labour0 * (1.0 + growth.labour) ** t
        supply[ki] = stock * user_cost
        if growth.productivity == "factor_augmenting":
            supply = supply * level
            tfp = np.ones(econ.n_act)
        else:
            tfp = np.full(econ.n_act, level)
        if sc.factor_supply is not None:
            supply = np.asarray(sc.factor_supply, dtype=float)
        if sc.productivity is not None:
            tfp = tfp * np.asarray(sc.productivity, dtype=float)
        changes = dict(factor_supply=supply, productivity=tfp)
        if sc.foreign_factor_share is None and (foreign.any() or row_share0.any()):
            share = e_t.zeros_if_none("fac_row_share", econ.n_fac).copy()
            share[ki] = np.where(stock > 0, foreign / np.maximum(stock, 1e-300), 0.0)
            changes["foreign_factor_share"] = share
        sc = sc.with_(**changes)

        res = solve_period(e_t, closure, sc, warm_start=warm, raise_on_fail=raise_on_fail)
        periods.append(res)
        stocks.append(stock.copy())
        foreigns.append(foreign.copy())
        warm = res

        # Real investment adds to the stock. The flow-onto-flow error this avoids
        # -- adding nominal saving to the capital SERVICE payment -- is the single
        # most common bug in a recursive dynamic.
        if growth.investment_deflator == "own_price":
            inv_w = e_t.inv_dem / max(float(e_t.inv_dem.sum()), 1e-12)
            p_inv = float(inv_w @ res.pqf)
            inv_real = res.savings / max(p_inv, 1e-12)
        else:
            inv_real = float(res.real["inv_real"])
        # FDI is part of this year's saving; the capital it buys is foreign-owned
        # and goes to the factor it was aimed at. The rest is domestic and is
        # shared by the allocation rule.
        real_per_nominal = inv_real / res.savings if abs(res.savings) > 1e-12 else 0.0
        fdi_real = np.array([float(sc.fdi.get(k, 0.0)) * closure.numeraire * real_per_nominal
                             for k in caps])
        domestic = inv_real - float(fdi_real.sum())
        weights = _allocation(growth.allocation, stock, res.factor_prices[ki])
        stock = (1.0 - dep) * stock + domestic * weights + fdi_real
        foreign = (1.0 - dep) * foreign + fdi_real

    single = len(caps) == 1
    shape = (lambda a: np.array(a)[:, 0]) if single else np.array
    return PathResult(years=years, periods=periods, econ=econ, growth=growth,
                      capital_stock=shape(stocks), foreign_capital=shape(foreigns),
                      capital_factors=caps)


def _allocation(rule, stock: np.ndarray, rent: np.ndarray) -> np.ndarray:
    """Shares of domestic investment going to each capital factor."""
    if stock.size == 1:
        return np.ones(1)
    weights = stock.copy()
    if rule != "proportional":
        mean = float(rent @ stock) / max(float(stock.sum()), 1e-300)
        weights = stock * (rent / max(mean, 1e-300)) ** float(rule)
    return weights / max(float(weights.sum()), 1e-300)
