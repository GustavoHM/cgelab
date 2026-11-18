"""Melitz firms: the closed forms against quadrature, published results and an independent solve.

``solve_melitz`` evaluates closed forms and solves nothing, so each check below reaches the same
number another way: by integrating the firm's problem over the Pareto draws, by a result from the
literature that the module does not use, or from ``tests/data/melitz.json``, which
``validation/melitz.jl`` wrote after solving the equilibrium from the primitives by quadrature.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import cgelab as cge

REF = json.loads((Path(__file__).parent / "data" / "melitz.json").read_text())
HEAVY_TAIL = cge.MelitzSpec(sigma=4.0, pareto_shape=3.5, entry_cost=2.0)
SCALE = cge.MelitzSpec(sigma=5.0, pareto_shape=6.5, productivity_min=2.0, labour=250.0,
                       entry_cost=2.0, domestic_fixed_cost=1.5, export_fixed_cost=2.0,
                       iceberg_cost=1.7)
SPECS = [cge.MelitzSpec(), HEAVY_TAIL, SCALE]


def _pdf(result, phi):
    s = result.spec
    return s.pareto_shape * s.productivity_min ** s.pareto_shape * phi ** (-s.pareto_shape - 1)


def _integral(result, function, cutoff):
    """The integral of ``function`` over the Pareto draws above ``cutoff``, in log productivity."""
    return quad(lambda x: function(np.exp(x)) * _pdf(result, np.exp(x)) * np.exp(x),
                np.log(cutoff), np.log(cutoff) + 90.0, epsabs=0, epsrel=1e-13, limit=400)[0]


@pytest.mark.parametrize("case", list(REF))
def test_every_result_matches_the_equilibrium_julia_solved_from_the_primitives(case):
    result = cge.solve_melitz(cge.MelitzSpec(**REF[case]["spec"]))
    for name, expected in REF[case]["result"].items():
        assert getattr(result, name) == pytest.approx(expected, rel=1e-10), name
    assert result.residual < 1e-11


def test_the_fixture_covers_every_reported_quantity():
    reported = {"domestic_cutoff", "export_cutoff", "entrants", "domestic_firms", "exporters",
                "price_index", "welfare", "domestic_revenue", "export_revenue",
                "average_domestic_productivity", "average_export_productivity",
                "real_wage", "export_share", "entry_probability", "domestic_share"}
    assert all(set(case["result"]) == reported for case in REF.values())


@pytest.mark.parametrize("spec", SPECS)
def test_free_entry_is_the_integral_of_operating_profits_over_the_pareto_draws(spec):
    result = cge.solve_melitz(spec)
    domestic = _integral(result, result.operating_profit, result.domestic_cutoff)
    export = _integral(result, lambda phi: result.operating_profit(phi, market="export"),
                       result.export_cutoff)
    assert domestic + export == pytest.approx(spec.entry_cost, rel=1e-10)


@pytest.mark.parametrize("spec", SPECS)
def test_the_price_index_is_the_integral_over_all_varieties(spec):
    result = cge.solve_melitz(spec)
    a = spec.sigma - 1
    price = lambda phi: spec.sigma / a / phi
    domestic = _integral(result, lambda phi: price(phi) ** -a, result.domestic_cutoff)
    imported = _integral(result, lambda phi: (spec.iceberg_cost * price(phi)) ** -a,
                         result.export_cutoff)
    assert result.entrants * (domestic + imported) == pytest.approx(
        result.price_index ** -a, rel=1e-10)


@pytest.mark.parametrize("spec", SPECS)
def test_firm_masses_revenues_and_labour_demand_integrate_to_the_aggregates(spec):
    result = cge.solve_melitz(spec)
    rho = (spec.sigma - 1) / spec.sigma
    survive = _integral(result, lambda phi: 1.0, result.domestic_cutoff)
    export = _integral(result, lambda phi: 1.0, result.export_cutoff)
    assert result.entrants * survive == pytest.approx(result.domestic_firms, rel=1e-10)
    assert result.entrants * export == pytest.approx(result.exporters, rel=1e-10)
    home = result.entrants * _integral(result, result.revenue, result.domestic_cutoff)
    abroad = result.entrants * _integral(
        result, lambda phi: result.revenue(phi, market="export"), result.export_cutoff)
    assert home == pytest.approx(result.domestic_revenue, rel=1e-10)
    assert abroad == pytest.approx(result.export_revenue, rel=1e-10)
    # A firm's variable labour is its output over its productivity: revenue over the markup.
    labour = result.entrants * (
        spec.entry_cost
        + _integral(result, lambda phi: spec.domestic_fixed_cost + rho * result.revenue(phi),
                    result.domestic_cutoff)
        + _integral(result, lambda phi: spec.export_fixed_cost
                    + rho * result.revenue(phi, market="export"), result.export_cutoff))
    assert labour == pytest.approx(spec.labour, rel=1e-10)


@pytest.mark.parametrize("spec", SPECS)
def test_average_productivity_is_the_ces_mean_of_the_firms_in_each_market(spec):
    result = cge.solve_melitz(spec)
    a = spec.sigma - 1
    for cutoff, average in ((result.domestic_cutoff, result.average_domestic_productivity),
                            (result.export_cutoff, result.average_export_productivity)):
        mean = (_integral(result, lambda phi: phi ** a, cutoff)
                / _integral(result, lambda phi: 1.0, cutoff)) ** (1 / a)
        assert average == pytest.approx(mean, rel=1e-10)


@pytest.mark.parametrize("spec", SPECS)
def test_welfare_per_worker_is_melitz_s_function_of_the_domestic_cutoff(spec):
    # Melitz (2003): W = rho (L / (sigma f)) ** (1 / (sigma - 1)) phi*, in the open economy too.
    result = cge.solve_melitz(spec)
    rho = (spec.sigma - 1) / spec.sigma
    per_worker = (rho * (spec.labour / (spec.sigma * spec.domestic_fixed_cost))
                  ** (1 / (spec.sigma - 1)) * result.domestic_cutoff)
    assert result.real_wage == pytest.approx(per_worker, rel=1e-12)
    assert result.welfare == pytest.approx(spec.labour * per_worker, rel=1e-12)


@pytest.mark.parametrize("spec", SPECS)
def test_the_price_index_is_that_of_the_representative_firm(spec):
    # Melitz (2003), section 6: P = M_t ** (1 / (1 - sigma)) / (rho phi_t), with phi_t the
    # average over domestic firms and the partner's exporters, the latter deflated by tau.
    result = cge.solve_melitz(spec)
    a = spec.sigma - 1
    combined = ((result.domestic_firms * result.average_domestic_productivity ** a
                 + result.exporters
                 * (result.average_export_productivity / spec.iceberg_cost) ** a)
                / result.varieties) ** (1 / a)
    assert result.price_index == pytest.approx(
        result.varieties ** (-1 / a) / (a / spec.sigma * combined), rel=1e-12)


@pytest.mark.parametrize("change", [{"iceberg_cost": 1.2}, {"iceberg_cost": 2.4},
                                    {"export_fixed_cost": 0.7}, {"export_fixed_cost": 2.5}])
def test_gains_from_trade_follow_arkolakis_costinot_and_rodriguez_clare(change):
    # W'/W = (lambda'/lambda) ** (-1 / k): the domestic expenditure share and the Pareto shape,
    # which is the trade elasticity, are sufficient for the welfare effect of a foreign shock.
    for spec in SPECS:
        base = cge.solve_melitz(spec)
        shocked = cge.solve_melitz(replace(spec, **change))
        assert shocked.welfare / base.welfare == pytest.approx(
            (shocked.domestic_share / base.domestic_share) ** (-1 / spec.pareto_shape),
            rel=1e-12)


@pytest.mark.parametrize("spec", SPECS)
def test_the_trade_elasticity_is_the_pareto_shape_not_sigma(spec):
    # Chaney (2008): with selection into exporting, trade flows respond to the iceberg cost
    # with elasticity k, whatever the elasticity of substitution.
    step = 1e-5
    ratio = []
    for tau in (spec.iceberg_cost * np.exp(-step), spec.iceberg_cost * np.exp(step)):
        result = cge.solve_melitz(replace(spec, iceberg_cost=tau))
        ratio.append(np.log(result.export_revenue / result.domestic_revenue))
    assert (ratio[1] - ratio[0]) / (2 * step) == pytest.approx(-spec.pareto_shape, rel=1e-8)


@pytest.mark.parametrize("spec", SPECS)
def test_prohibitive_trade_costs_give_the_closed_economy(spec):
    # The closed economy of Melitz (2003) with Pareto draws: the cutoff from free entry and the
    # zero-profit condition, and every entrant's revenue spent at home.
    a, k = spec.sigma - 1, spec.pareto_shape
    result = cge.solve_melitz(replace(spec, iceberg_cost=1e8))
    cutoff = spec.productivity_min * (
        a * spec.domestic_fixed_cost / ((k - a) * spec.entry_cost)) ** (1 / k)
    assert result.domestic_cutoff == pytest.approx(cutoff, rel=1e-12)
    assert result.domestic_share == pytest.approx(1.0, abs=1e-12)
    assert result.exporters == pytest.approx(0.0, abs=1e-12)
    assert result.domestic_firms == pytest.approx(
        spec.labour * (k - a) / (spec.sigma * k * spec.domestic_fixed_cost), rel=1e-12)
    assert cge.solve_melitz(spec).welfare > result.welfare


def test_cutoffs_select_and_the_accounts_close():
    result = cge.solve_melitz()
    s = result.spec
    assert result.operating_profit(result.domestic_cutoff) == pytest.approx(0, abs=1e-12)
    assert result.operating_profit(result.export_cutoff, market="export") == pytest.approx(
        0, abs=1e-12)
    assert result.operating_profit(result.domestic_cutoff * 0.99) < 0
    assert result.operating_profit(result.export_cutoff * 1.01, market="export") > 0
    assert result.domestic_revenue + result.export_revenue == pytest.approx(s.labour, rel=1e-12)
    assert set(result.residuals) == {"domestic_zero_profit", "export_zero_profit", "free_entry",
                                     "labour", "revenue"}
    assert result.residual == max(map(abs, result.residuals.values())) < 1e-12
    assert result.export_share == pytest.approx(
        (result.domestic_cutoff / result.export_cutoff) ** s.pareto_shape, rel=1e-12)


@pytest.mark.parametrize("change", [{"iceberg_cost": 1.2}, {"export_fixed_cost": 0.7}])
def test_lower_trade_costs_expand_export_selection_and_welfare(change):
    base = cge.solve_melitz()
    counterfactual = cge.solve_melitz(replace(base.spec, **change))
    assert counterfactual.export_cutoff < base.export_cutoff
    assert counterfactual.export_share > base.export_share
    assert counterfactual.domestic_cutoff > base.domestic_cutoff
    assert counterfactual.domestic_firms < base.domestic_firms
    assert counterfactual.price_index < base.price_index
    assert counterfactual.welfare > base.welfare


def test_only_interior_selection_is_accepted():
    with pytest.raises(ValueError, match="above one"):
        cge.solve_melitz(cge.MelitzSpec(sigma=1.0))
    with pytest.raises(ValueError, match="pareto_shape"):
        cge.solve_melitz(cge.MelitzSpec(pareto_shape=3.0))
    with pytest.raises(ValueError, match="at least one"):
        cge.solve_melitz(cge.MelitzSpec(iceberg_cost=0.9))
    with pytest.raises(ValueError, match="all-exporters"):
        cge.solve_melitz(cge.MelitzSpec(iceberg_cost=1.0))
    with pytest.raises(ValueError, match="all-exporters"):
        cge.solve_melitz(cge.MelitzSpec(iceberg_cost=1.2, export_fixed_cost=0.5))
    with pytest.raises(ValueError, match="all-entrants-produce"):
        cge.solve_melitz(cge.MelitzSpec(entry_cost=10.0))
    with pytest.raises(ValueError, match="finite and positive"):
        cge.solve_melitz(cge.MelitzSpec(labour=0.0))
    with pytest.raises(ValueError, match="iceberg_cost must be finite"):
        cge.solve_melitz(cge.MelitzSpec(iceberg_cost=float("inf")))


def test_firm_revenue_checks_its_market_and_productivity():
    result = cge.solve_melitz()
    with pytest.raises(ValueError, match="market"):
        result.revenue(1.0, market="foreign")
    with pytest.raises(ValueError, match="market"):
        result.operating_profit(1.0, market="foreign")
    with pytest.raises(ValueError, match="productivity"):
        result.revenue(0.0)
    both = result.revenue(np.array([1.5, 3.0]))
    assert both[1] / both[0] == pytest.approx(2.0 ** (result.spec.sigma - 1), rel=1e-12)
    assert result.revenue(2.0, market="export") == pytest.approx(
        result.revenue(2.0) * result.spec.iceberg_cost ** (1 - result.spec.sigma), rel=1e-12)


def test_the_melitz_api_is_public():
    assert {"MelitzSpec", "MelitzResult", "solve_melitz"} <= set(cge.__all__)
