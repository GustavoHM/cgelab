"""cgelab -- a general toolkit for SAM-based computable general equilibrium models.

A model is assembled from three separable objects, and keeping them separable is
the point: the :class:`~cgelab.sam.SAM` says what the base year WAS, the
:class:`~cgelab.spec.ModelSpec` says what is ASSUMED about substitution and
functional form, and the :class:`~cgelab.closure.Closure` says which balances are
imposed and WHO adjusts. Results can turn on the last two, so both travel with
the results instead of living in a branch inside a solver.

Quick start::

    import cgelab as cge
    from cgelab.examples import toy

    sam, econ = toy.build()
    print(cge.report(econ))                      # base year, Walras, homogeneity

    base = cge.solve_period(econ)
    shock = cge.solve_period(econ, scenario=cge.Scenario(
        name="tax", product_tax=econ.ptax * 1.10))
    print(shock.summary())

The numeraire factor's market is deliberately left out of the solved system, so
its residual is an independent check on the accounting rather than an identity
the solver was told to satisfy.
"""

from .calibrate import Economy, calibrate
from .closure import Closure, LabourMarket
from .decompose import path_decomposition, scenario_vector, subtotals, welfare_decomposition
from .diagnostics import base_year_reproduction, homogeneity_test, report, walras_check
from .emissions import Emissions, add_emissions, emissions
from .physical import PhysicalUnits, add_physical_units, physical, physical_coefficient
from .export import to_jump
from .har import read_har, write_har
from .dynamics import Growth, PathResult, run_path
from .functions import (armington, ces_intensity, ces_price, cet, cet_domestic_price,
                        cobb_douglas_price)
from .behaviour import Behaviour, PersonSurvey, transition_probabilities
from .micro import (MicroResult, Survey, account_sources, atkinson, fgt, fgt_decomposition,
                    ge_decomposition, generalized_entropy, gini, inequality,
                    mean_log_deviation, microsimulate, poverty, theil)
from .mobility import FactorMobility
from .multipliers import (Multipliers, accounting_multipliers, fixed_price_multipliers,
                          leontief_multipliers, linkages, structural_paths)
from .results import (factor_demand_by_activity, government_account, national_accounts,
                      price_indices, real_gdp, sectoral_path)
from .sam import SAM, BalanceError
from .scenario import Scenario, consumer_wedge
from .sensitivity import (Normal, SensitivityResult, Triangular, Uniform, sensitivity,
                          spec_sweep, stroud_points)
from .solve import PeriodResult, factor_intensity, solve_period, value_added_price
from .spec import AccountLayout, ModelSpec
from .welfare import compensating_variation, equivalent_variation, ev_by_group

__version__ = "0.5.0"

__all__ = [
    "SAM", "BalanceError",
    "ModelSpec", "AccountLayout",
    "calibrate", "Economy",
    "Closure", "LabourMarket", "Scenario", "consumer_wedge",
    "solve_period", "PeriodResult", "value_added_price", "factor_intensity",
    "run_path", "PathResult", "Growth",
    "FactorMobility",
    "Survey", "MicroResult", "microsimulate", "account_sources", "inequality", "poverty",
    "PersonSurvey", "Behaviour", "transition_probabilities",
    "gini", "fgt", "generalized_entropy", "theil", "mean_log_deviation", "atkinson",
    "ge_decomposition", "fgt_decomposition",
    "equivalent_variation", "ev_by_group", "compensating_variation",
    "base_year_reproduction", "homogeneity_test", "walras_check", "report",
    "to_jump", "read_har", "write_har",
    "Emissions", "add_emissions", "emissions",
    "PhysicalUnits", "add_physical_units", "physical", "physical_coefficient",
    "factor_demand_by_activity", "real_gdp", "sectoral_path",
    "national_accounts", "price_indices", "government_account",
    "Multipliers", "accounting_multipliers", "fixed_price_multipliers", "leontief_multipliers",
    "linkages", "structural_paths",
    "subtotals", "path_decomposition", "scenario_vector", "welfare_decomposition",
    "sensitivity", "spec_sweep", "stroud_points", "SensitivityResult",
    "Uniform", "Normal", "Triangular",
    "ces_price", "ces_intensity", "cobb_douglas_price", "armington", "cet", "cet_domestic_price",
    "__version__",
]
