"""Two regions, one origin nest. The base year is the SAM, not the solver's memory.

A single labelled region is the same model as the legacy solver: the origin CES
has one variety, and its price is that variety's price.
"""

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import regions


def test_one_labelled_region_matches_the_legacy_solver():
    import pandas as pd
    from cgelab.examples import toy
    sam, econ = toy.build()
    legacy = cge.solve_period(econ)
    keep = {econ.layout.government, econ.layout.investment, econ.layout.rest_of_world}

    def ren(name):
        return name if name in keep else f"br:{name}"

    rows = [(ren(r), ren(c), v) for r, c, v in sam.frame.itertuples(index=False)]
    sam2 = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]), name="br")
    spec = cge.ModelSpec(
        regions=("br",), numeraire_factor="br:f_lab", factors=("f_lab", "f_cap"),
        sigma_va=econ.spec.sigma_va, sigma_armington=econ.spec.sigma_armington,
        margins=False, value_added=econ.spec.value_added)
    econ2 = cge.calibrate(sam2, spec)
    regional = cge.solve_period(econ2)
    assert regional.converged and abs(regional.walras) < 1e-8
    for name in econ.acts:
        assert regional.X[econ2.activity(f"br:{name}")] == pytest.approx(
            legacy.X[econ.activity(name)], abs=1e-8)


def test_the_base_year_reproduces_both_regions():
    _, econ = regions.build()
    period = cge.solve_period(econ)
    assert period.converged
    assert abs(period.walras) < 1e-8
    assert period.X[econ.activity("r1:a")] == pytest.approx(100.0, abs=1e-6)
    assert period.X[econ.activity("r2:a")] == pytest.approx(100.0, abs=1e-6)
    assert period.Q[econ.commodity("r1:c")] == pytest.approx(100.0, abs=1e-6)
    assert period.Q[econ.commodity("r2:c")] == pytest.approx(100.0, abs=1e-6)
    assert period.factor_prices[econ.factor("r2:f_cap")] == pytest.approx(1.0, abs=1e-6)
    # Domestic sales of each origin are output minus the base-year export share.
    # Destination-specific bilateral cells have their own reconciliation tests.
    r1 = econ.commodity("r1:c")
    assert period.Q[r1] - period.exports[r1] == pytest.approx(90.0, abs=1e-6)


def test_a_productivity_gain_raises_own_output_and_the_other_regions_purchases():
    _, econ = regions.build()
    base = cge.solve_period(econ)
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("r1:a")] = 1.2
    shock = cge.solve_period(econ, scenario=cge.Scenario(productivity=tfp))
    assert shock.converged and abs(shock.walras) < 1e-8
    assert shock.X[econ.activity("r1:a")] > base.X[econ.activity("r1:a")]
    # bilateral[origin, destination, commodity]
    r1, r2 = 0, 1
    assert shock.bilateral[r1, r2, 0] > base.bilateral[r1, r2, 0]


def test_a_transfer_between_regions_keeps_national_walras():
    _, econ = regions.build()
    base = cge.solve_period(econ)
    # Household order follows the spec: r1, then r2.
    moved = cge.solve_period(econ, scenario=cge.Scenario(transfers=np.array([-10.0, 10.0])))
    assert moved.converged and abs(moved.walras) < 1e-8
    assert moved.hh_income[econ.household("r2:hh")] > base.hh_income[econ.household("r2:hh")]
    assert moved.hh_income[econ.household("r1:hh")] < base.hh_income[econ.household("r1:hh")]


def test_swapping_region_names_swaps_a_shock():
    """The base year is symmetric, so the swap is checked on a shock that is not."""
    _, left = regions.build()
    _, right = regions.build(rename=("r1", "r2"))
    tfp_l = np.ones(left.n_act)
    tfp_l[left.activity("r1:a")] = 1.2
    tfp_r = np.ones(right.n_act)
    tfp_r[right.activity("r2:a")] = 1.2
    a = cge.solve_period(left, scenario=cge.Scenario(productivity=tfp_l))
    b = cge.solve_period(right, scenario=cge.Scenario(productivity=tfp_r))
    assert a.X[left.activity("r1:a")] == pytest.approx(b.X[right.activity("r2:a")], abs=1e-6)
    assert a.X[left.activity("r2:a")] == pytest.approx(b.X[right.activity("r1:a")], abs=1e-6)
    # Nominal income depends on which region's factor is the numeraire. Real
    # absorption does not.
    assert a.real["hh_real"] == pytest.approx(b.real["hh_real"], abs=1e-6)
    assert a.X[left.activity("r1:a")] != pytest.approx(a.X[left.activity("r2:a")], rel=1e-3)


def test_a_missing_origin_is_named():
    import pandas as pd
    from cgelab.sam import SAM
    rows = [e for e in regions.ENTRIES if "r2:c" not in (e[0], e[1])]
    sam = SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    with pytest.raises(KeyError, match="r2:c"):
        cge.calibrate(sam, regions.spec(), check_balance=False)


def test_non_default_closures_are_refused_by_name():
    _, econ = regions.build()
    with pytest.raises(ValueError, match="labour"):
        cge.solve_period(econ, closure=cge.Closure(labour="fixed_real_wage"))
    with pytest.raises(ValueError, match="to_jump"):
        cge.to_jump(econ)
    with pytest.raises(ValueError, match="run_path"):
        cge.run_path(econ, years=[1])


@pytest.mark.parametrize("export_rule", ["fixed_shares", "cet"])
def test_regions_importing_different_shares_reproduce_the_base_year(export_rule):
    sam, econ = regions.build(regions.spec(export_rule=export_rule, sigma_cet=3.0),
                              asymmetric_imports=True)
    result = cge.solve_period(econ)
    assert abs(result.walras) < 1e-8
    np.testing.assert_allclose(result.factor_prices, 1.0, atol=1e-9)
    assert result.exchange_rate == pytest.approx(1.0)
    np.testing.assert_allclose(result.imports, [15.0, 5.0], atol=1e-8)
    np.testing.assert_allclose(result.hh_commodity_spend,
                               sam.block(list(econ.coms), list(econ.hhs)).sum(axis=0),
                               atol=1e-8)
    home = econ.regional.dom_share_dest[:2, 0]
    np.testing.assert_allclose(home, [(55 * 90 / 105 + 20 * 90 / 95) / 75,
                                      (20 * 90 / 105 + 50 * 90 / 95) / 70])
