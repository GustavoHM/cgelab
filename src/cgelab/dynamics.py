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
    #: The labour factor that grows, or a group of ``ModelSpec.labour_segments``.
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
    #: The stock of each pool of a capital factor that has pools
    #: (``calibrate(factor_pools=)``): ``{factor: DataFrame}``, years by pools.
    #: Sector-specific capital is a pool per activity. Empty without pools.
    capital_by_pool: dict = field(repr=False, default_factory=dict)

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
        """Output and prices of every account along the path. See :func:`cgelab.results.sectoral_path`."""
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
    years = list(years)
    state = _PathState(econ, closure, growth, raise_on_fail=raise_on_fail)
    for t, year in enumerate(years):
        e_t = state.economy_at(economy, year, t)
        sc = scenario(year, t) if callable(scenario) else (scenario or Scenario())
        res, sc_full = state.solve(e_t, sc, t)
        state.advance(e_t, res, sc_full)
    return state.result(years)


class _PathState:
    """What carries from one year of a path to the next, and the step between them.

    ``solve`` builds the year's scenario from the growth block and the state and
    solves the period, without changing the state, so that a calibration can try
    several productivities for one year; ``advance`` then accumulates capital
    from the solution that was kept.
    """

    def __init__(self, econ: Economy, closure: Closure, growth: Growth, *,
                 raise_on_fail: bool = True):
        if econ.regional is not None:
            local = [k for k in growth.capital_factors + [growth.labour_factor] if ":" not in k]
            if local:
                raise ValueError("regional run_path needs full 'region:factor' names in "
                                 f"Growth.capital_factor and labour_factor, got {local}")
        self.econ, self.closure, self.growth = econ, closure, growth
        self.raise_on_fail = raise_on_fail
        self.caps = growth.capital_factors
        self.ki = np.array([econ.factor(k) for k in self.caps])
        # A group of labour segments grows as a whole: every segment, and so its
        # labour force, at the same rate.
        self.li = (np.array(econ.segments[growth.labour_factor].index)
                   if growth.labour_factor in econ.segments
                   else econ.factor(growth.labour_factor))
        self.dep = np.array([growth.depreciation_of(k) for k in self.caps])
        self.user_cost = growth.discount + self.dep
        # The stocks that accumulate: one per capital factor, or one per pool
        # of a capital factor that has pools -- sector-specific capital is a
        # pool per activity. `owner` says which capital factor each belongs to.
        owner, base = [], []
        for c, i in enumerate(self.ki):
            mob = econ.mobility.get(int(i))
            flows = [econ.fac_supply[i]] if mob is None else list(mob.endowment0)
            owner += [c] * len(flows)
            base += flows
        self.owner = np.array(owner, dtype=int)
        self.units = np.array(base, dtype=float) / self.user_cost[self.owner]
        self.pooled = {c: np.where(self.owner == c)[0] for c, i in enumerate(self.ki)
                       if int(i) in econ.mobility}
        self.row_share0 = econ.zeros_if_none("fac_row_share", econ.n_fac)
        self.foreign = self.stock * self.row_share0[self.ki]
        self.labour0 = np.array(econ.fac_supply[self.li], dtype=float)
        self.periods, self.stocks, self.foreigns, self.unit_stocks = [], [], [], []
        self.warm = None
        # Putty-clay: the old vintage is the capital factor, a stock per
        # activity; the new vintage is last year's investment, one stock.
        self.vintage = econ.vintage
        self.new_stock, self.new_stocks = 0.0, []
        if self.vintage is not None:
            v = self.vintage
            if self.caps != [v.old]:
                raise ValueError(
                    f"with vintages, Growth.capital_factor must be the old vintage, "
                    f"{v.old!r}, alone: the new one, {v.new!r}, is each year's investment, "
                    f"and a second stock would need a share of it. Got {self.caps}.")
            if self.row_share0[[v.i_old, v.i_new]].any():
                raise ValueError("vintages with foreign-owned capital are not available yet: "
                                 "the SAM pays capital income abroad")
            self.new_stock = float(econ.fac_supply[v.i_new]) / float(self.user_cost[0])

    @property
    def stock(self) -> np.ndarray:
        """The stock of each capital factor, its pools summed."""
        return np.bincount(self.owner, weights=self.units, minlength=len(self.caps))

    def economy_at(self, economy, year, t) -> Economy:
        econ = self.econ
        e_t = economy(year, t) if economy is not None else econ
        if e_t.factors != econ.factors or e_t.coms != econ.coms or e_t.acts != econ.acts:
            raise ValueError(f"economy({year!r}, {t}) returned an Economy with different "
                             f"accounts; build it from the base with dataclasses.replace")
        if self.vintage is not None and e_t.vintage is not self.vintage:
            # The old vintage carries the techniques it was installed with.
            import dataclasses
            e_t = dataclasses.replace(e_t, vintage=self.vintage)
        return e_t

    def solve(self, e_t: Economy, sc: Scenario, t: int, *, warm=None,
              raise_on_fail: bool | None = None):
        """The period at index ``t`` under ``sc``; returns it and the scenario it solved."""
        econ, growth = self.econ, self.growth
        unknown = [k for k in sc.fdi if k not in self.caps]
        if unknown:
            raise ValueError(f"FDI into {unknown}, which Growth does not accumulate as capital "
                             f"(capital factors: {self.caps})")
        level = (1.0 + growth.tfp) ** t
        supply = e_t.fac_supply.copy()
        supply[self.li] = self.labour0 * (1.0 + growth.labour) ** t
        supply[self.ki] = self.stock * self.user_cost
        if self.vintage is not None:
            if sc.fdi:
                raise ValueError("vintages with FDI are not available yet")
            supply[self.vintage.i_new] = self.new_stock * self.user_cost[0]
        if growth.productivity == "factor_augmenting":
            supply = supply * level
            tfp = np.ones(econ.n_act)
        else:
            tfp = np.full(econ.n_act, level)
        changes = {}
        if sc.factor_supply is not None:
            supply = np.asarray(sc.factor_supply, dtype=float)
        elif self.pooled:
            # Each pool's services come from its own stock. A scenario that sets
            # a factor's supply, or its pools, overrides them, as it overrides
            # any accumulated supply.
            scale = level if growth.productivity == "factor_augmenting" else 1.0
            pools = dict(sc.pool_supply or {})
            for c, members in self.pooled.items():
                if self.caps[c] in pools:
                    supply[self.ki[c]] = float(np.sum(pools[self.caps[c]]))
                else:
                    pools[self.caps[c]] = self.units[members] * self.user_cost[c] * scale
            changes["pool_supply"] = pools
        if sc.productivity is not None:
            tfp = tfp * np.asarray(sc.productivity, dtype=float)
        changes.update(factor_supply=supply, productivity=tfp)
        if sc.foreign_factor_share is None and (self.foreign.any() or self.row_share0.any()):
            share = e_t.zeros_if_none("fac_row_share", econ.n_fac).copy()
            share[self.ki] = np.where(self.stock > 0,
                                      self.foreign / np.maximum(self.stock, 1e-300), 0.0)
            changes["foreign_factor_share"] = share
        sc_full = sc.with_(**changes)
        res = solve_period(e_t, self.closure, sc_full,
                           warm_start=self.warm if warm is None else warm,
                           raise_on_fail=(self.raise_on_fail if raise_on_fail is None
                                          else raise_on_fail))
        return res, sc_full

    def advance(self, e_t: Economy, res: PeriodResult, sc: Scenario) -> None:
        """Keep the year's solution and accumulate capital from its investment."""
        growth, closure = self.growth, self.closure
        self.periods.append(res)
        self.stocks.append(self.stock + self.new_stock)
        self.foreigns.append(self.foreign.copy())
        self.unit_stocks.append(self.units.copy())
        self.new_stocks.append(self.new_stock)
        self.warm = res

        # Real investment adds to the stock. The flow-onto-flow error this avoids
        # -- adding nominal saving to the capital SERVICE payment -- is the single
        # most common bug in a recursive dynamic.
        if growth.investment_deflator == "own_price":
            inv_w = e_t.inv_dem / max(float(e_t.inv_dem.sum()), 1e-12)
            p_inv = float(inv_w @ res.pqf)
            inv_real = res.savings / max(p_inv, 1e-12)
        else:
            inv_real = float(res.real["inv_real"])
        if self.vintage is not None:
            # The new capital each activity used this year is installed there:
            # next year it is old capital of that activity, and this year's
            # investment is next year's new capital.
            v = self.vintage
            use = res.factor_demand_by_activity[v.i_new]
            total = float(use.sum())
            installed = self.new_stock * use / total if total > 0 else np.zeros_like(use)
            members = self.pooled[0]
            where = [self.econ.activity(a) for a in self.econ.mobility[v.i_old].pools]
            self.units[members] = (1.0 - self.dep[0]) * (self.units[members] + installed[where])
            self.new_stock = inv_real
            self.vintage = v.installed(res.factor_prices_by_activity, res.vintage_value_added)
            return
        # FDI is part of this year's saving; the capital it buys is foreign-owned
        # and goes to the factor it was aimed at. The rest is domestic and is
        # shared by the allocation rule.
        real_per_nominal = inv_real / res.savings if abs(res.savings) > 1e-12 else 0.0
        fdi_real = np.array([float(sc.fdi.get(k, 0.0)) * closure.numeraire * real_per_nominal
                             for k in self.caps])
        domestic = inv_real - float(fdi_real.sum())
        # A pool's rent is its price: at one pool per activity, the price the
        # activity pays for the factor.
        rent = res.factor_prices[self.ki][self.owner].astype(float)
        for c, members in self.pooled.items():
            rent[members] = res.factor_pools[self.caps[c]]["price"]
        weights = _allocation(growth.allocation, self.units, rent)
        # FDI aimed at a factor with pools is shared among them as domestic
        # investment is.
        within = np.bincount(self.owner, weights=weights, minlength=len(self.caps))
        fdi_units = fdi_real[self.owner] * np.where(
            within[self.owner] > 0, weights / np.maximum(within[self.owner], 1e-300),
            0.0)
        self.units = ((1.0 - self.dep[self.owner]) * self.units + domestic * weights
                      + fdi_units)
        self.foreign = (1.0 - self.dep) * self.foreign + fdi_real

    def result(self, years) -> "PathResult":
        single = len(self.caps) == 1
        shape = (lambda a: np.array(a)[:, 0]) if single else np.array
        units = np.array(self.unit_stocks).reshape(len(self.unit_stocks), -1)
        by_pool = {self.caps[c]: pd.DataFrame(
                       units[:, members], index=list(years),
                       columns=list(self.econ.mobility[int(self.ki[c])].pools))
                   for c, members in self.pooled.items()}
        if self.vintage is not None:
            new = self.vintage.new
            by_pool[new] = pd.DataFrame({new: self.new_stocks}, index=list(years))
        return PathResult(years=list(years), periods=self.periods, econ=self.econ,
                          growth=self.growth, capital_stock=shape(self.stocks),
                          foreign_capital=shape(self.foreigns), capital_factors=self.caps,
                          capital_by_pool=by_pool)


def _allocation(rule, stock: np.ndarray, rent: np.ndarray) -> np.ndarray:
    """Shares of domestic investment going to each capital factor."""
    if stock.size == 1:
        return np.ones(1)
    weights = stock.copy()
    if rule != "proportional":
        mean = float(rent @ stock) / max(float(stock.sum()), 1e-300)
        weights = stock * (rent / max(mean, 1e-300)) ** float(rule)
    return weights / max(float(weights.sum()), 1e-300)
