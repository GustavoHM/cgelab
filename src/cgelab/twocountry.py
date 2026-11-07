"""Two countries that trade with each other, each a national model.

Hosoe, Gasawa and Hashimoto's ``twocge`` (chapter 10.3): each country is a complete national
economy, with its own currency, government, saving and balance of payments, and what one exports
of a good the other imports. ``solve_two_countries`` solves each country with
:func:`cgelab.solve_period`, the world (dollar) prices of its exports and imports given, and finds
the world prices at which the two countries' trade clears good by good. Every feature of the
national model, taxes and closures included, holds in each country.

With both balances of payments and every market but one, the last market clears by Walras' law,
so one condition is free. As in ``twocge``, one country's exchange rate is held at its base-year
value in place of the redundant market, which sets the level of world prices. That choice is not
neutral when foreign saving is fixed in dollars, as in ``twocge``: the dollar is the unit of the
foreign saving, so holding one exchange rate or the other changes what it is worth. Say which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import root

from .calibrate import Economy
from .closure import Closure
from .scenario import Scenario
from .solve import PeriodResult, solve_period

__all__ = ["TwoCountry", "solve_two_countries"]


@dataclass
class TwoCountry:
    """Two countries solved together.

    ``periods`` holds each country's :class:`PeriodResult`, by name. ``world_prices`` are the
    dollar prices of each good as exported by each country (rows: goods; columns: exporter),
    which are the importer's import prices. ``trade`` is each country's exports of each good and
    the other's imports of it, which agree at the solution.
    """

    periods: dict
    world_prices: pd.DataFrame
    trade: pd.DataFrame
    iterations: int = 0
    residual: float = 0.0
    economies: dict = field(default_factory=dict, repr=False)


def _shared_goods(econ_a: Economy, econ_b: Economy) -> list:
    if list(econ_a.coms) != list(econ_b.coms):
        raise ValueError(f"the two countries must have the same commodities, in the same order: "
                         f"{list(econ_a.coms)} against {list(econ_b.coms)}")
    for name, e in (("first", econ_a), ("second", econ_b)):
        if e.regional is not None:
            raise ValueError(f"the {name} country is a regional model; each country is national")
        if e.spec.export_rule != "cet":
            raise ValueError(f"the {name} country has fixed export shares; a country's exports "
                             f"must answer to their world price (export_rule='cet')")
    return list(econ_a.coms)


def solve_two_countries(econ_a: Economy, econ_b: Economy, *,
                        closure_a: Closure | None = None, closure_b: Closure | None = None,
                        scenario_a: Scenario | None = None, scenario_b: Scenario | None = None,
                        names: tuple[str, str] = ("a", "b"), numeraire: int = 1,
                        tol: float = 1e-12) -> TwoCountry:
    """Solve two national economies that trade only with each other.

    Each country's foreign saving should be fixed in foreign currency
    (``Closure(foreign_saving_unit="foreign_currency")``), so that both balances of payments are
    in dollars, and the two should mirror each other at the base year. ``numeraire`` (0 or 1) is
    the country whose exchange rate stays at its base-year value. The scenarios may set anything
    but ``world_price`` and ``export_price``, which the solver sets.
    """
    goods = _shared_goods(econ_a, econ_b)
    n = len(goods)
    econs = (econ_a, econ_b)
    closures = (closure_a or Closure(foreign_saving_unit="foreign_currency"),
                closure_b or Closure(foreign_saving_unit="foreign_currency"))
    scenarios = (scenario_a or Scenario(), scenario_b or Scenario())
    for k, sc in enumerate(scenarios):
        if sc.world_price is not None or sc.export_price is not None:
            raise ValueError(f"scenario for {names[k]!r} sets world prices; the two-country "
                             f"solver sets them from the other country's trade")
    if numeraire not in (0, 1):
        raise ValueError("numeraire is 0 or 1, the country whose exchange rate is held")
    # The base year must be a two-country equilibrium: each country's exports are the other's
    # imports, good by good.
    gap = np.max(np.abs(econ_a.exp0 - econ_b.imp0)) + np.max(np.abs(econ_b.exp0 - econ_a.imp0))
    if gap > 1e-8 * max(1.0, float(econ_a.exp0.max())):
        raise ValueError("at the base year one country's exports of a good are not the other's "
                         "imports of it; the two SAMs do not describe trade with each other")
    warm: list = [None, None]
    memo: dict = {"n": 0}

    def solve(x):
        p = np.exp(x).reshape(2, n)          # p[k, i]: dollar price of good i exported by k
        out = []
        for k in range(2):
            sc = scenarios[k].with_(export_price=p[k], world_price=p[1 - k])
            period = solve_period(econs[k], closures[k], sc, warm_start=warm[k])
            warm[k] = period
            out.append(period)
        return p, out

    def residual(x):
        memo["n"] += 1
        p, (pa, pb) = solve(x)
        res = np.concatenate([np.log(pa.exports / pb.imports), np.log(pb.exports / pa.imports)])
        # The last market of the non-numeraire country's exports is Walras' redundant one; the
        # numeraire's exchange rate takes its place.
        drop = (1 - numeraire) * n + n - 1
        er = (pa, pb)[numeraire].exchange_rate / closures[numeraire].exchange_rate
        res[drop] = np.log(er)
        memo["last"] = (p, pa, pb)
        return res

    sol = root(residual, np.zeros(2 * n), method="hybr", tol=tol)
    final = residual(sol.x)
    worst = float(np.max(np.abs(final)))
    if not worst <= max(tol * 100, 1e-9):
        raise RuntimeError(f"solve_two_countries did not converge: worst residual {worst:.3g} "
                           f"after {memo['n']} pairs of period solves ({sol.message})")
    p, pa, pb = memo["last"]
    world = pd.DataFrame(p.T, index=goods, columns=list(names))
    trade = pd.DataFrame({f"{names[0]} exports": pa.exports, f"{names[1]} imports": pb.imports,
                          f"{names[1]} exports": pb.exports, f"{names[0]} imports": pa.imports},
                         index=goods)
    return TwoCountry(periods={names[0]: pa, names[1]: pb}, world_prices=world, trade=trade,
                      iterations=memo["n"], residual=worst,
                      economies={names[0]: econ_a, names[1]: econ_b})
