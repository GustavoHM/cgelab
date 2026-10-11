"""Two regions, one origin nest. The base year is the SAM, not the solver's memory.

A single labelled region is the same model as the legacy solver: the origin CES
has one variety, and its price is that variety's price.
"""

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import regions


def test_one_labelled_region_matches_the_legacy_solver():
    import pandas as pd
    from cgelab.examples import toy
    sam, econ = toy.build()
    legacy = ck.solve_period(econ)
    keep = {econ.layout.government, econ.layout.investment, econ.layout.rest_of_world}

    def ren(name):
        return name if name in keep else f"br:{name}"

    rows = [(ren(r), ren(c), v) for r, c, v in sam.frame.itertuples(index=False)]
    sam2 = ck.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]), name="br")
    spec = ck.ModelSpec(
        regions=("br",), numeraire_factor="br:f_lab", factors=("f_lab", "f_cap"),
        sigma_va=econ.spec.sigma_va, sigma_armington=econ.spec.sigma_armington,
        margins=False, value_added=econ.spec.value_added)
    econ2 = ck.calibrate(sam2, spec)
    regional = ck.solve_period(econ2)
    assert regional.converged and abs(regional.walras) < 1e-8
    for name in econ.acts:
        assert regional.X[econ2.activity(f"br:{name}")] == pytest.approx(
            legacy.X[econ.activity(name)], abs=1e-8)


def test_the_base_year_reproduces_both_regions():
    _, econ = regions.build()
    period = ck.solve_period(econ)
    assert period.converged
    assert abs(period.walras) < 1e-8
    assert period.X[econ.activity("r1:a")] == pytest.approx(100.0, abs=1e-6)
    assert period.X[econ.activity("r2:a")] == pytest.approx(100.0, abs=1e-6)
    assert period.Q[econ.commodity("r1:c")] == pytest.approx(100.0, abs=1e-6)
    assert period.Q[econ.commodity("r2:c")] == pytest.approx(100.0, abs=1e-6)
    assert period.factor_prices[econ.factor("r2:f_cap")] == pytest.approx(1.0, abs=1e-6)
    # Domestic sales of each origin are output minus the base-year export share.
    # Bilateral cells of the SAM are not reproduced: buyers share one national mix.
    r1 = econ.commodity("r1:c")
    assert period.Q[r1] - period.exports[r1] == pytest.approx(90.0, abs=1e-6)


def test_a_productivity_gain_raises_own_output_and_the_other_regions_purchases():
    _, econ = regions.build()
    base = ck.solve_period(econ)
    tfp = np.ones(econ.n_act)
    tfp[econ.activity("r1:a")] = 1.2
    shock = ck.solve_period(econ, scenario=ck.Scenario(productivity=tfp))
    assert shock.converged and abs(shock.walras) < 1e-8
    assert shock.X[econ.activity("r1:a")] > base.X[econ.activity("r1:a")]
    # bilateral[origin, destination, commodity]
    r1, r2 = 0, 1
    assert shock.bilateral[r1, r2, 0] > base.bilateral[r1, r2, 0]


def test_a_transfer_between_regions_keeps_national_walras():
    _, econ = regions.build()
    base = ck.solve_period(econ)
    # Household order follows the spec: r1, then r2.
    moved = ck.solve_period(econ, scenario=ck.Scenario(transfers=np.array([-10.0, 10.0])))
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
    a = ck.solve_period(left, scenario=ck.Scenario(productivity=tfp_l))
    b = ck.solve_period(right, scenario=ck.Scenario(productivity=tfp_r))
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
        ck.calibrate(sam, regions.spec(), check_balance=False)


def test_non_default_closures_are_refused_by_name():
    _, econ = regions.build()
    with pytest.raises(ValueError, match="labour"):
        ck.solve_period(econ, closure=ck.Closure(labour="fixed_real_wage"))
    with pytest.raises(ValueError, match="to_jump"):
        ck.to_jump(econ)
    with pytest.raises(ValueError, match="run_path"):
        ck.run_path(econ, years=[1])
