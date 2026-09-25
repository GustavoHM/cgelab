"""Welfare measurement, and the two ways it is read wrongly."""

import numpy as np
import pytest

import cgekit as ck
from cgekit.examples import toy


def _paths(econ, scenario=None, years=range(2021, 2031)):
    return ck.run_path(econ, years=years, scenario=scenario)


def test_a_path_against_itself_has_zero_welfare_change():
    _, econ = toy.build()
    ref = _paths(econ)
    ev = ck.equivalent_variation(ref, ref)
    assert ev.ev.abs().max() < 1e-12
    assert ev.ev_base.abs().max() < 1e-12


def test_a_consumer_wedge_makes_everyone_worse_off():
    _, econ = toy.build()
    W = ck.consumer_wedge(econ, {"c_ind": 0.06})
    ref = _paths(econ)
    alt = _paths(econ, ck.Scenario(name="wedge", consumer_wedge=W))
    ev = ck.equivalent_variation(ref, alt)
    assert (ev.ev < 0).all()


def test_the_incidence_follows_the_recycling_rule():
    """Under capital recycling the capital-rich household bears less of the wedge."""
    _, econ = toy.build()
    W = ck.consumer_wedge(econ, {"c_ind": 0.06})
    ref = _paths(econ)
    cap = ck.equivalent_variation(ref, _paths(
        econ, ck.Scenario(consumer_wedge=W, wedge_revenue_to="capital")))
    lump = ck.equivalent_variation(ref, _paths(
        econ, ck.Scenario(consumer_wedge=W, wedge_revenue_to="lumpsum")))
    rich = cap[cap.account == "hh_1"].ev_pct_account.mean()
    rich_lump = lump[lump.account == "hh_1"].ev_pct_account.mean()
    assert rich > rich_lump


def test_group_aggregation_is_immune_to_migration_between_accounts():
    """Moving income between IDENTICAL twin accounts is composition, not welfare.

    The per-account measure reports it as a large welfare change in both
    directions; the group measure, which sums real incomes before dividing, sees
    the zero that is actually there. Any distributional claim must use the
    second, and the per-account column is named so that no call site can take it
    for the first by accident.
    """
    _, econ = toy.build_with_twin_accounts()
    a, b = econ.household("hh_2a"), econ.household("hh_2b")

    shifted = econ.hh_facshare.copy()
    for f in range(econ.n_fac):
        total = shifted[a, f] + shifted[b, f]
        shifted[a, f], shifted[b, f] = 0.6 * total, 0.4 * total

    ref = _paths(econ)
    alt = _paths(econ, ck.Scenario(name="migration", factor_ownership=shifted))

    ev = ck.equivalent_variation(ref, alt)
    per_account = ev[ev.account.isin(["hh_2a", "hh_2b"])].ev_pct_account.abs().max()
    assert per_account > 5.0                       # composition, reported as welfare

    groups = {"hh_1": "hh_1", "hh_2a": "hh_2", "hh_2b": "hh_2"}
    grouped = ck.ev_by_group(ev, groups)
    assert grouped.ev_pct.abs().max() < 1e-9       # the zero that is actually there


def test_every_account_must_be_assigned_to_a_group():
    _, econ = toy.build_with_twin_accounts()
    ref = _paths(econ, years=range(2021, 2023))
    ev = ck.equivalent_variation(ref, ref)
    with pytest.raises(KeyError, match="no group given"):
        ck.ev_by_group(ev, {"hh_1": "all"})


def test_compensating_and_equivalent_variation_agree_for_a_small_shock():
    _, econ = toy.build()
    W = ck.consumer_wedge(econ, {"c_ind": 0.002})
    ref = _paths(econ, years=range(2021, 2024))
    alt = _paths(econ, ck.Scenario(consumer_wedge=W), years=range(2021, 2024))
    ev = ck.equivalent_variation(ref, alt).set_index(["year", "account"]).ev
    cv = ck.compensating_variation(ref, alt).set_index(["year", "account"]).cv
    assert np.allclose(ev, cv, rtol=5e-3)
