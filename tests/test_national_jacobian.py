"""The exact Jacobian of the national residual against central finite differences."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
import cgelab.solve as national
from cgelab import national_jacobian
from cgelab.examples import hosoe, toy


@pytest.fixture
def captured(monkeypatch):
    """The residual and Jacobian functions solve_period hands to its root finder."""
    seen = {}
    original = national.root

    def spy(fun, x0, *args, **kwargs):
        seen.update(residuals=fun, jac=kwargs.get("jac"))
        return original(fun, x0, *args, **kwargs)

    monkeypatch.setattr(national, "root", spy)
    return seen


def _central(f, z, h=2e-4):
    """Central differences, two steps extrapolated: the error is of order h**4.

    The residual stops its price fixed point at 1e-14, so a plain central
    difference with a step small enough for its own h**2 error would divide
    that noise by the step and be no better than 1e-9.
    """
    def at(step):
        cols = []
        for k in range(z.size):
            e = np.zeros(z.size)
            e[k] = step
            cols.append((np.asarray(f(z + e)) - np.asarray(f(z - e))) / (2 * step))
        return np.array(cols).T
    return (4.0 * at(h / 2) - at(h)) / 3.0


TFP = np.array([1.1, 1.0])
# Two pools of labour, one mostly in agriculture and one mostly in industry.
POOLS = pd.DataFrame([[30.0, 10.0], [10.0, 60.0]], index=["rural", "urban"],
                     columns=["a_agr", "a_ind"])


def _sluggish(form="cet", pools=None, **spec):
    return cge.calibrate(toy.build()[0],
                         toy.spec(numeraire_factor="f_cap", factor_mobility={"f_lab": 2.0},
                                  factor_mobility_form=form, **spec),
                         factor_pools=None if pools is None else {"f_lab": pools})


CASES = {
    "standard": (lambda: toy.build()[1], cge.Closure(), cge.Scenario(productivity=TFP)),
    "margins": (lambda: toy.build_with_margins()[1], cge.Closure(),
                cge.Scenario(productivity=np.array([1.1, 1.0, 1.0]))),
    "cobb_douglas": (lambda: toy.build(model_spec=toy.spec(value_added="cobb_douglas"))[1],
                     cge.Closure(), cge.Scenario(productivity=TFP)),
    # Fixed proportions in one activity, the nest in the other.
    "leontief_activity": (
        lambda: dataclasses.replace(toy.build()[1], leontief=np.array([True, False])),
        cge.Closure(), cge.Scenario(productivity=TFP, world_price=np.array([1.0, 1.2]))),
    "own_account": (lambda: toy.build_with_own_production()[1], cge.Closure(),
                    cge.Scenario(productivity=np.array([1.1, 1.0, 1.0]),
                                 activity_scale={"a_own": 1.2})),
    "hosoe_saving_rate": (lambda: hosoe.build()[1], hosoe.CLOSURE,
                          cge.Scenario(tariff=np.zeros(2))),
    "trade_taxes": (lambda: toy.build_with_trade_taxes()[1], cge.Closure(),
                    cge.Scenario(tariff=np.zeros(2), output_tax=np.array([0.01, 0.03]))),
    "export_tax_cet": (lambda: toy.build_with_export_tax()[1], cge.Closure(),
                       cge.Scenario(export_price=np.array([1.2, 0.9]),
                                    export_tax=np.array([0.0, 0.1]))),
    "institutions": (lambda: toy.build_with_institutions()[1], cge.Closure(),
                     cge.Scenario(productivity=TFP, transfers=np.array([2.0, -1.0]),
                                  fdi={"f_cap": 3.0}, government_absorbs=1.0)),
    "nominal_government": (lambda: toy.build_with_institutions()[1],
                           cge.Closure(government="nominal_fixed"),
                           cge.Scenario(productivity=TFP)),
    "saving_target_spending": (
        lambda: toy.build_with_institutions()[1],
        cge.Closure(government_target="saving", government_adjusts="spending"),
        cge.Scenario(productivity=TFP)),
    "fixed_investment": (lambda: toy.build()[1], cge.Closure(investment="fixed_real"),
                         cge.Scenario(investment_level=60.0)),
    "fixed_rate": (lambda: toy.build()[1], cge.Closure(foreign="fixed_rate"),
                   cge.Scenario(productivity=TFP)),
    "foreign_currency": (lambda: toy.build()[1],
                         cge.Closure(foreign_saving_unit="foreign_currency"),
                         cge.Scenario(productivity=TFP)),
    "combined_closures": (
        lambda: toy.build_with_institutions()[1],
        cge.Closure(government="saving_rate", investment="fixed_real", foreign="fixed_rate"),
        cge.Scenario(productivity=TFP, investment_level=60.0)),
    "les": (lambda: toy.build(model_spec=toy.spec(household_demand="les", frisch=-2.0))[1],
            cge.Closure(), cge.Scenario(productivity=TFP)),
    "elastic_supply": (
        lambda: toy.build(model_spec=toy.spec(numeraire_factor="f_cap"))[1],
        cge.Closure(labour="elastic_supply", labour_supply_elasticity=0.4),
        cge.Scenario(productivity=TFP)),
    "investment_mix_numeraire": (
        lambda: toy.build()[1], cge.Closure(numeraire=2.0),
        cge.Scenario(productivity=TFP, investment_mix=np.array([0.4, 0.6]),
                     product_tax_firm=np.array([0.02, 0.06]))),
    "zero_wedges": (lambda: toy.build()[1], cge.Closure(),
                    cge.Scenario(productivity=TFP, consumer_wedge=np.zeros((2, 2)),
                                 government_wedge=np.zeros((2, 2)))),
    # Large-country trade: a world price per commodity on a curve.
    "trade_curves": (
        lambda: toy.build(model_spec=toy.spec(export_rule="cet", export_demand=2.0,
                                              import_supply=3.0))[1],
        cge.Closure(), cge.Scenario(productivity=TFP, export_price=np.array([1.1, 1.0]))),
    "import_curve_fixed_shares": (
        lambda: toy.build(model_spec=toy.spec(import_supply={"c_agr": 1.5}))[1],
        cge.Closure(investment="fixed_real"),
        cge.Scenario(world_price=np.array([1.2, 1.0]), investment_level=55.0)),
    # A factor that is not perfectly mobile: its price in each activity.
    "mobility_cet": (_sluggish, cge.Closure(), cge.Scenario(productivity=TFP)),
    "mobility_logit": (lambda: _sluggish("logit"), cge.Closure(),
                       cge.Scenario(productivity=TFP, world_price=np.array([1.0, 1.2]))),
    "mobility_pools": (lambda: _sluggish(pools=POOLS), cge.Closure(foreign="fixed_rate"),
                       cge.Scenario(productivity=TFP,
                                    pool_supply={"f_lab": np.array([42.0, 68.0])})),
    "mobility_logit_pools_curves": (
        lambda: _sluggish("logit", POOLS, export_rule="cet", export_demand=2.0,
                          import_supply=3.0),
        cge.Closure(investment="fixed_real", government="saving_rate"),
        cge.Scenario(productivity=TFP, investment_level=55.0)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_exact_jacobian_matches_finite_differences(captured, case):
    build, closure, scenario = CASES[case]
    econ = build()
    assert national_jacobian.unsupported(econ, closure, scenario) is None
    result = cge.solve_period(econ, closure, scenario)
    f, jac = captured["residuals"], captured["jac"]
    assert jac is not None
    for z in (result.z, result.z + np.linspace(0.02, -0.03, result.z.size)):
        exact, numeric = jac(z), _central(f, z)
        assert exact.shape == numeric.shape == (z.size, z.size)
        assert np.max(np.abs(exact - numeric)) < 1e-10 * max(np.max(np.abs(numeric)), 1.0)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_exact_jacobian_reaches_the_same_equilibrium(monkeypatch, case):
    build, closure, scenario = CASES[case]
    econ = build()
    exact = cge.solve_period(econ, closure, scenario)
    monkeypatch.setattr(national_jacobian, "unsupported", lambda *args: "switched off")
    numeric = cge.solve_period(econ, closure, scenario)
    assert exact.converged and numeric.converged
    assert exact.residual < 1e-13 and abs(exact.walras) < 1e-12
    for name in ("factor_prices", "pq", "Q", "X", "absorption", "imports", "exports",
                 "gov_quantity", "investment_quantity", "hh_income"):
        np.testing.assert_allclose(getattr(exact, name), getattr(numeric, name),
                                   rtol=1e-13, atol=1e-13, err_msg=name)
    assert exact.exchange_rate == pytest.approx(numeric.exchange_rate, rel=1e-13)
    assert exact.savings == pytest.approx(numeric.savings, rel=1e-13)


def test_the_base_year_is_reproduced():
    for econ, closure in ((toy.build()[1], cge.Closure()), (hosoe.build()[1], hosoe.CLOSURE),
                          (_sluggish(pools=POOLS), cge.Closure())):
        result = cge.solve_period(econ, closure)
        assert result.converged and abs(result.walras) < 1e-12
        np.testing.assert_allclose(result.X, econ.X0, rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# What the derivation does not cover keeps finite differences, by name.
# ---------------------------------------------------------------------------

def _segmented():
    return toy.build_with_segmented_labour()[1]


def _priced_carbon():
    return cge.add_emissions(toy.build()[1], intermediate=np.array([[1.0, 2.0], [0.0, 3.0]]))


def _vintages():
    return cge.calibrate(cge.split_vintages(toy.build()[0], 0.1),
                         toy.spec(factors=("f_lab", "f_cap", "f_cap_new"),
                                  vintages={"f_cap": "f_cap_new"}))


def _backstop():
    return cge.calibrate(toy.build()[0], toy.spec(),
                         backstops={"a_new": cge.Backstop("c_ind", like="a_ind", markup=1.25)})


INST = lambda: toy.build_with_institutions()[1]                      # noqa: E731
PLAIN = lambda: toy.build()[1]                                       # noqa: E731
FALLBACKS = {
    # option outside the derivation: (economy, closure, scenario, the name reported)
    "least_cost": (lambda: toy.build_with_technologies(
        model_spec=toy.spec(least_cost=["c_ind"]))[1], cge.Closure(),
        cge.Scenario(productivity=np.array([1.0, 0.9, 1.1])), "ModelSpec.least_cost"),
    "energy_nest": (lambda: toy.build(model_spec=toy.spec(energy=["c_ind"]))[1], cge.Closure(),
                    cge.Scenario(productivity=TFP), "ModelSpec.energy"),
    "asymmetric_armington": (
        lambda: toy.build(model_spec=toy.spec(sigma_armington_falling=4.0))[1], cge.Closure(),
        cge.Scenario(world_price=np.array([0.9, 1.1])), "ModelSpec.sigma_armington_falling"),
    "cde": (lambda: toy.build(model_spec=toy.spec(household_demand="cde",
                                                  cde_substitution=0.5))[1], cge.Closure(),
            cge.Scenario(productivity=TFP), "ModelSpec.household_demand='cde'"),
    "leisure": (lambda: toy.build(model_spec=toy.spec(numeraire_factor="f_cap",
                                                      leisure_ratio=0.5))[1], cge.Closure(),
                cge.Scenario(productivity=TFP), "ModelSpec.leisure_ratio"),
    "monopoly": (lambda: toy.build(model_spec=toy.spec(monopoly=["c_ind"]))[1], cge.Closure(),
                 cge.Scenario(productivity=TFP), "ModelSpec.monopoly"),
    "fixed_costs": (lambda: toy.build(model_spec=toy.spec(fixed_costs={"a_ind": 0.2}))[1],
                    cge.Closure(), cge.Scenario(productivity=TFP), "ModelSpec.fixed_costs"),
    "labour_segments": (_segmented, cge.Closure(), cge.Scenario(productivity=TFP),
                        "ModelSpec.labour_segments"),
    "vintages": (_vintages, cge.Closure(), cge.Scenario(productivity=TFP),
                 "ModelSpec.vintages"),
    "backstop": (_backstop, cge.Closure(),
                 cge.Scenario(productivity=np.array([1.1, 1.0, 1.0])),
                 "Economy.least_cost"),
    "direct_tax_lever": (INST, cge.Closure(government_target="saving",
                                           government_adjusts="direct_tax"),
                         cge.Scenario(productivity=TFP),
                         "Closure.government_adjusts='direct_tax'"),
    "revenue_target": (INST, cge.Closure(government_target="revenue",
                                         government_adjusts="product_tax"),
                       cge.Scenario(productivity=TFP), "Closure.government_target='revenue'"),
    "fixed_real_wage": (lambda: toy.build(model_spec=toy.spec(numeraire_factor="f_cap"))[1],
                        cge.Closure(labour="fixed_real_wage"), cge.Scenario(productivity=TFP),
                        "Closure.labour='fixed_real_wage'"),
    "wage_curve": (lambda: toy.build(model_spec=toy.spec(numeraire_factor="f_cap"))[1],
                   cge.Closure(labour="wage_curve", natural_unemployment=0.08),
                   cge.Scenario(productivity=TFP), "Closure.labour='wage_curve'"),
    "wage_floor": (lambda: toy.build(model_spec=toy.spec(numeraire_factor="f_cap"))[1],
                   cge.Closure(labour="wage_floor"), cge.Scenario(productivity=TFP),
                   "Closure.labour='wage_floor'"),
    "consumer_wedge": (PLAIN, cge.Closure(),
                       cge.Scenario(consumer_wedge=np.array([[0.1, 0.0], [0.1, 0.0]])),
                       "Scenario.consumer_wedge"),
    "government_wedge": (PLAIN, cge.Closure(),
                         cge.Scenario(government_wedge=np.array([[0.0, -0.05], [0.0, -0.05]])),
                         "Scenario.government_wedge"),
    "carbon_price": (_priced_carbon, cge.Closure(), cge.Scenario(carbon_price=0.05),
                     "Scenario.carbon_price"),
    "capacity": (PLAIN, cge.Closure(), cge.Scenario(capacity={"a_ind": 190.0}),
                 "Scenario.capacity"),
    "import_quota": (PLAIN, cge.Closure(), cge.Scenario(import_quota={"c_ind": 22.0}),
                     "Scenario.import_quota"),
    "price_ceiling": (PLAIN, cge.Closure(),
                      cge.Scenario(productivity=np.array([0.8, 1.0]),
                                   price_ceiling={"c_agr": 1.02}), "Scenario.price_ceiling"),
    "price_floor": (PLAIN, cge.Closure(),
                    cge.Scenario(productivity=np.array([1.3, 1.0]),
                                 price_floor={"c_agr": 0.871}), "Scenario.price_floor"),
}


@pytest.mark.parametrize("option", sorted(FALLBACKS))
def test_an_option_outside_the_derivation_keeps_finite_differences(captured, option):
    build, closure, scenario, name = FALLBACKS[option]
    econ = build()
    assert national_jacobian.unsupported(econ, closure, scenario) == name
    result = cge.solve_period(econ, closure, scenario)
    assert result.converged
    # The root finder ran, and without a Jacobian function.
    assert "residuals" in captured and captured["jac"] is None


@pytest.mark.parametrize("cls, label, kwargs", [
    (cge.Scenario, "Scenario", {}),
    (cge.Closure, "Closure", {}),
])
def test_an_option_added_later_is_declined_until_it_is_listed(cls, label, kwargs):
    """The gate is a list of what is derived: a new field is outside it by construction."""
    later = dataclasses.make_dataclass("Later", [("new_option", object, None)], bases=(cls,))
    econ = toy.build()[1]
    args = {"Scenario": lambda v: (econ, cge.Closure(), later(new_option=v)),
            "Closure": lambda v: (econ, later(new_option=v), cge.Scenario())}[label]
    assert national_jacobian.unsupported(*args(None)) is None        # off: nothing to derive
    assert national_jacobian.unsupported(*args(0.3)) == f"{label}.new_option"


def test_a_field_added_to_the_economy_or_the_spec_is_declined():
    econ = toy.build()[1]
    spec_later = dataclasses.make_dataclass(
        "SpecLater", [("new_block", object, None)], bases=(cge.ModelSpec,))
    fields = {f.name: getattr(econ.spec, f.name) for f in dataclasses.fields(econ.spec)}
    on = dataclasses.replace(econ, spec=spec_later(**fields, new_block={"c_ind": 1.0}))
    off = dataclasses.replace(econ, spec=spec_later(**fields))
    assert national_jacobian.unsupported(off, cge.Closure(), cge.Scenario()) is None
    assert national_jacobian.unsupported(on, cge.Closure(), cge.Scenario()) == "ModelSpec.new_block"
    econ_later = dataclasses.make_dataclass(
        "EconomyLater", [("new_block", object, None)], bases=(type(econ),))
    values = {f.name: getattr(econ, f.name) for f in dataclasses.fields(econ)}
    assert national_jacobian.unsupported(
        econ_later(**values, new_block=np.ones(2)), cge.Closure(), cge.Scenario()
    ) == "Economy.new_block"
    assert national_jacobian.unsupported(
        econ_later(**values), cge.Closure(), cge.Scenario()) is None


def test_every_option_field_is_either_derived_or_declined_at_its_default():
    """No field without a default escapes the gate: there would be no way to switch it off."""
    for cls, known, values in ((cge.ModelSpec, national_jacobian._SPEC,
                                national_jacobian._SPEC_VALUES),
                               (cge.Closure, national_jacobian._CLOSURE,
                                national_jacobian._CLOSURE_VALUES),
                               (cge.Scenario, national_jacobian._SCENARIO, {})):
        names = {f.name for f in dataclasses.fields(cls)}
        # Every name the gate lists is a field: a renamed option cannot linger as covered.
        assert set(known) | set(values) <= names, (set(known) | set(values)) - names
    listed = national_jacobian._ECONOMY
    assert listed <= {f.name for f in dataclasses.fields(type(toy.build()[1]))}
