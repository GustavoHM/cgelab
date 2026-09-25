"""cgekit -- a general toolkit for SAM-based computable general equilibrium models.

A model is assembled from three separable objects, and keeping them separable is
the point: the :class:`~cgekit.sam.SAM` says what the base year WAS, the
:class:`~cgekit.spec.ModelSpec` says what is ASSUMED about substitution and
functional form, and the :class:`~cgekit.closure.Closure` says which balances are
imposed and WHO adjusts. Results can turn on the last two, so both travel with
the results instead of living in a branch inside a solver.

Quick start::

    import cgekit as ck
    from cgekit.examples import toy

    sam, econ = toy.build()
    print(ck.report(econ))                       # base year, Walras, homogeneity

    base = ck.solve_period(econ)
    shock = ck.solve_period(econ, scenario=ck.Scenario(
        name="tax", product_tax=econ.ptax * 1.10))
    print(shock.summary())

The numeraire factor's market is deliberately left out of the solved system, so
its residual is an independent check on the accounting rather than an identity
the solver was told to satisfy.
"""

from .calibrate import Economy, calibrate
from .closure import Closure
from .diagnostics import base_year_reproduction, homogeneity_test, report, walras_check
from .export import to_jump
from .dynamics import Growth, PathResult, run_path
from .functions import (armington, ces_intensity, ces_price, cet, cet_domestic_price,
                        cobb_douglas_price)
from .sam import SAM, BalanceError
from .scenario import Scenario, consumer_wedge
from .solve import PeriodResult, factor_intensity, solve_period, value_added_price
from .spec import AccountLayout, ModelSpec
from .welfare import compensating_variation, equivalent_variation, ev_by_group

__version__ = "0.2.0"

__all__ = [
    "SAM", "BalanceError",
    "ModelSpec", "AccountLayout",
    "calibrate", "Economy",
    "Closure", "Scenario", "consumer_wedge",
    "solve_period", "PeriodResult", "value_added_price", "factor_intensity",
    "run_path", "PathResult", "Growth",
    "equivalent_variation", "ev_by_group", "compensating_variation",
    "base_year_reproduction", "homogeneity_test", "walras_check", "report",
    "to_jump",
    "ces_price", "ces_intensity", "cobb_douglas_price", "armington", "cet", "cet_domestic_price",
    "__version__",
]
