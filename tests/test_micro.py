"""Microsimulation: inequality and poverty measures, and the survey link to the model.

The measures are checked against answers worked out on paper and against a
brute-force formula. The link is checked against the model itself: a survey
that is the SAM, one household per account, must give back the model's incomes
and real incomes, and a reconciled survey must add up to the model's accounts,
source by source.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab import micro
from cgelab.examples import toy

RNG = np.random.default_rng(20261019)


# -- measures -----------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 5, 50])
def test_gini_of_the_first_integers(n):
    assert micro.gini(np.arange(1, n + 1)) == pytest.approx((n - 1) / (3 * n), rel=1e-14)


def test_gini_of_two_points():
    # Shares p and 1 − p at incomes a < b: G = p (1 − p) (b − a) / μ.
    a, b, p = 2.0, 7.0, 0.3
    mu = p * a + (1 - p) * b
    assert micro.gini([a, b], [p, 1 - p]) == pytest.approx(p * (1 - p) * (b - a) / mu, rel=1e-14)


def test_gini_is_the_mean_absolute_difference():
    x, w = RNG.lognormal(size=40), RNG.uniform(0.5, 3.0, size=40)
    brute = (w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :])).sum()
    brute /= 2.0 * w.sum() ** 2 * (w @ x / w.sum())
    assert micro.gini(x, w) == pytest.approx(brute, rel=1e-12)


def test_fgt_on_paper():
    x = [1.0, 2.0, 3.0, 4.0]
    assert micro.fgt(x, 3.0, 0) == 0.5                       # 3 is not below the line
    assert micro.fgt(x, 3.0, 1) == pytest.approx((2 / 3 + 1 / 3) / 4, rel=1e-15)
    assert micro.fgt(x, 3.0, 2) == pytest.approx((4 / 9 + 1 / 9) / 4, rel=1e-15)


def test_generalized_entropy_and_atkinson_on_two_points():
    x, p = np.array([1.0, 4.0]), np.array([0.5, 0.5])
    mu = 2.5
    mld = np.log(mu) - 0.5 * (np.log(1.0) + np.log(4.0))
    theil_t = 0.5 * (1 / mu * np.log(1 / mu) + 4 / mu * np.log(4 / mu))
    ge2 = 0.5 * (0.5 * (1 - mu) ** 2 + 0.5 * (4 - mu) ** 2) / mu ** 2
    assert micro.mean_log_deviation(x, p) == pytest.approx(mld, rel=1e-14)
    assert micro.theil(x, p) == pytest.approx(theil_t, rel=1e-14)
    assert micro.generalized_entropy(x, 2.0, p) == pytest.approx(ge2, rel=1e-14)
    assert micro.atkinson(x, 1.0, p) == pytest.approx(1 - 2.0 / mu, rel=1e-14)   # geometric mean 2
    assert micro.atkinson(x, 2.0, p) == pytest.approx(1 - 1.6 / mu, rel=1e-14)   # harmonic mean 1.6
    assert micro.atkinson(x, 0.0, p) == 0.0


def test_identities_between_the_indices():
    x, w = RNG.lognormal(size=30), RNG.uniform(0.5, 2.0, size=30)
    # A(1) = 1 − exp(−MLD); GE(α) is continuous at 0 and 1.
    assert micro.atkinson(x, 1.0, w) == pytest.approx(1 - np.exp(-micro.mean_log_deviation(x, w)),
                                                      rel=1e-13)
    assert micro.generalized_entropy(x, 1e-7, w) == pytest.approx(
        micro.mean_log_deviation(x, w), rel=1e-5)
    assert micro.generalized_entropy(x, 1 + 1e-7, w) == pytest.approx(micro.theil(x, w), rel=1e-5)


@pytest.mark.parametrize("measure", [
    lambda x, w: micro.gini(x, w), lambda x, w: micro.fgt(x, 1.2, 0, w),
    lambda x, w: micro.fgt(x, 1.2, 2, w), lambda x, w: micro.theil(x, w),
    lambda x, w: micro.mean_log_deviation(x, w), lambda x, w: micro.atkinson(x, 0.5, w),
    lambda x, w: micro.atkinson(x, 2.0, w)])
def test_integer_weights_are_replication(measure):
    x = RNG.lognormal(size=12)
    w = RNG.integers(1, 5, size=12)
    assert measure(x, w) == pytest.approx(measure(np.repeat(x, w), None), rel=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
def test_ge_decomposition_closes(alpha):
    x, w = RNG.lognormal(size=60), RNG.uniform(0.5, 2.0, size=60)
    groups = RNG.choice(["a", "b", "c"], size=60)
    d = micro.ge_decomposition(x, alpha, groups, w)
    assert d.at["within", "contribution"] + d.at["between", "contribution"] == pytest.approx(
        d.at["total", "contribution"], rel=1e-12)
    assert d.at["total", "contribution"] == pytest.approx(
        micro.generalized_entropy(x, alpha, w), rel=1e-14)


@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_fgt_decomposition_closes(alpha):
    x, w = RNG.lognormal(size=60), RNG.uniform(0.5, 2.0, size=60)
    groups = RNG.choice(["a", "b", "c"], size=60)
    d = micro.fgt_decomposition(x, 1.0, alpha, groups, w)
    assert d.loc[["a", "b", "c"], "contribution"].sum() == pytest.approx(
        d.at["total", "index"], rel=1e-14)


def test_measures_refuse_what_they_cannot_measure():
    with pytest.raises(ValueError, match="positive incomes"):
        micro.theil([0.0, 1.0])
    with pytest.raises(ValueError, match="positive incomes"):
        micro.atkinson([0.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="positive mean"):
        micro.gini([0.0, 0.0])
    with pytest.raises(ValueError, match="poverty line"):
        micro.fgt([1.0], 0.0)
    with pytest.raises(ValueError, match="weights"):
        micro.gini([1.0, 2.0], [1.0, -1.0])


# -- the link to the model ---------------------------------------------------

def dual(real_wage=1.1):
    return cge.Closure(labour_markets={"labour": cge.LabourMarket(
        rigid={"f_for": real_wage}, informal="f_inf")})


CASES = {
    "institutions_tax": (toy.build_with_institutions, cge.Closure(),
                         lambda e: cge.Scenario(product_tax=e.ptax * 1.3)),
    "institutions_wedge": (toy.build_with_institutions, cge.Closure(),
                           lambda e: cge.Scenario(
                               consumer_wedge=cge.consumer_wedge(e, {"c_ind": 0.1}),
                               wedge_revenue_to="lumpsum",
                               transfers=np.array([2.0, 0.0]), government_absorbs=2.0)),
    "institutions_target": (toy.build_with_institutions,
                            cge.Closure(government_target="revenue",
                                        government_adjusts="direct_tax",
                                        investment="fixed_real"),
                            lambda e: cge.Scenario(product_tax=e.ptax * 0.5)),
    "segmented_dual": (toy.build_with_segmented_labour, dual(),
                       lambda e: cge.Scenario(productivity=np.array([1.05, 1.0]))),
    "fixed_real_wage": (lambda: toy.build(model_spec=toy.spec(numeraire_factor="f_cap")),
                        cge.Closure(labour="fixed_real_wage"),
                        lambda e: cge.Scenario(product_tax=e.ptax * 1.5)),
}


def sam_as_survey(period):
    """One household per account, weight 1 and size 1, its incomes the SAM's."""
    A = micro.account_sources(period)
    cols = [c for c in A.columns if ":" not in c and np.any(np.abs(A[c]) > 1e-12)]
    frame = A[cols].copy()
    frame.columns = [f"y_{c}" for c in cols]
    frame.insert(0, "size", 1.0)
    frame.insert(0, "weight", 1.0)
    frame.insert(0, "group", A.index)
    return micro.Survey(frame.reset_index(drop=True), sources={f"y_{c}": c for c in cols})


@pytest.mark.parametrize("case", list(CASES))
def test_account_sources_add_up_to_household_income(case):
    build, closure, scenario = CASES[case]
    _, econ = build()
    res = cge.solve_period(econ, closure, scenario(econ))
    A = micro.account_sources(res)
    whole = [c for c in A.columns if ":" not in c]
    np.testing.assert_allclose(A[whole].sum(axis=1), res.hh_income, rtol=1e-12, atol=1e-12)
    for f in econ.factors:
        parts = [c for c in A.columns if c.startswith(f"{f}:")]
        np.testing.assert_allclose(A[parts].sum(axis=1), A[f], rtol=1e-12, atol=1e-12)


def test_the_numeraire_is_paid_its_supply():
    # A transfer nobody pays for manufactures income, and the numeraire's market,
    # left to Walras' law, shows it. The household accounts still add up,
    # because the numeraire is paid what it supplies, as the solver pays it.
    _, econ = toy.build_with_institutions()
    res = cge.solve_period(econ, scenario=cge.Scenario(transfers=np.array([2.0, 0.0])))
    assert abs(res.walras) > 1e-3
    A = micro.account_sources(res)
    whole = [c for c in A.columns if ":" not in c]
    np.testing.assert_allclose(A[whole].sum(axis=1), res.hh_income, rtol=1e-12)


@pytest.mark.parametrize("case", list(CASES))
def test_a_survey_that_is_the_sam_gives_back_the_model(case):
    build, closure, scenario = CASES[case]
    _, econ = build()
    ref = cge.solve_period(econ, closure)
    alt = cge.solve_period(econ, closure, scenario(econ))
    # One household per account cannot be reweighted: it is the account.
    res = micro.microsimulate(sam_as_survey(ref), ref, alt, reweight=False)
    f = res.frame
    rate = lambda p: p.hh_direct_tax / p.hh_income
    np.testing.assert_allclose(f["disposable_ref"], ref.hh_income * (1 - rate(ref)), rtol=1e-12)
    np.testing.assert_allclose(f["disposable_alt"], alt.hh_income * (1 - rate(alt)), rtol=1e-12)
    # The household's own index is the model's, so real income moves as the model's.
    model = lambda p: p.hh_income * (1 - rate(p)) / p.real["hh_price_index"]
    np.testing.assert_allclose(f["real_alt"] / f["real_ref"], model(alt) / model(ref),
                               rtol=1e-12)
    np.testing.assert_allclose(res.reconciliation["factor"], 1.0, rtol=1e-12)


@pytest.mark.parametrize("case", ["segmented_dual"])
def test_a_reconciled_survey_adds_up_to_the_model(case):
    build, closure, scenario = CASES[case]
    _, econ = build()
    ref = cge.solve_period(econ, dual(1.0))           # the base year, which the survey describes
    alt = cge.solve_period(econ, closure, scenario(econ))
    res = micro.microsimulate(toy.survey(), ref, alt)
    A_ref, A_alt = micro.account_sources(ref), micro.account_sources(alt)
    w = res.frame["weight"].to_numpy()
    w_alt = res.frame["weight_alt"].to_numpy()
    for h, s in itertools.product(econ.hhs, ["f_cap", "f_for", "f_inf"]):
        m = (res.frame["group"] == h).to_numpy()
        assert w[m] @ res.income_ref.loc[m, s] == pytest.approx(A_ref.at[h, s], rel=1e-12)
        assert w_alt[m] @ res.income_alt.loc[m, s] == pytest.approx(A_alt.at[h, s], rel=1e-12)
    r = res.reconciliation.set_index(["group", "source"])
    assert r.at[("hh_1", "f_for"), "factor"] == pytest.approx(60.0 / 51.0, rel=1e-14)
    assert r.at[("hh_2", "f_cap"), "factor"] == pytest.approx(15.0 / 8.75, rel=1e-14)


def test_without_reconciliation_only_relative_changes_apply():
    _, econ = toy.build_with_segmented_labour()
    ref, alt = cge.solve_period(econ, dual(1.0)), cge.solve_period(econ, dual(1.1))
    res = micro.microsimulate(toy.survey(), ref, alt, reconcile=False, reweight=False)
    assert res.reconciliation.empty
    A_ref, A_alt = micro.account_sources(ref), micro.account_sources(alt)
    s = toy.survey().frame
    m = (s["group"] == "hh_2").to_numpy()
    np.testing.assert_allclose(res.income_alt.loc[m, "f_for"],
                               s.loc[m, "formal"] * A_alt.at["hh_2", "f_for"]
                               / A_ref.at["hh_2", "f_for"], rtol=1e-14)


def test_a_new_flow_is_shared_per_person_within_the_account():
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ)
    alt = cge.solve_period(econ, scenario=cge.Scenario(transfers=np.array([0.0, 6.0]),
                                                       government_absorbs=6.0))
    res = micro.microsimulate(toy.survey(), ref, alt)
    f = res.frame
    m = (f["group"] == "hh_2").to_numpy()
    new = res.income_alt.loc[m, "new_flows"].to_numpy()
    persons = (f.loc[m, "weight_alt"] * f.loc[m, "size"]).to_numpy()
    assert float(f.loc[m, "weight_alt"] @ new) == pytest.approx(6.0, rel=1e-12)
    np.testing.assert_allclose(new / f.loc[m, "size"], 6.0 / persons.sum(), rtol=1e-12)
    assert np.all(res.income_alt.loc[~m, "new_flows"] == 0.0)


def test_the_household_price_index():
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ)
    alt = cge.solve_period(econ, scenario=cge.Scenario(product_tax=econ.ptax * 2.0))
    geo = micro.microsimulate(toy.survey(), ref, alt).frame["price_index"]
    las = micro.microsimulate(toy.survey(), ref, alt, price_index="laspeyres").frame["price_index"]
    assert np.all(las >= geo - 1e-15)                  # arithmetic mean over geometric mean
    # A household that buys only food pays food's price change.
    s = toy.survey()
    s.frame.loc[0, "goods"] = 0.0
    only_food = micro.microsimulate(s, ref, alt).frame["price_index"].iloc[0]
    p_ref = micro._effective_prices(ref)[0, econ.commodity("c_agr")]
    p_alt = micro._effective_prices(alt)[0, econ.commodity("c_agr")]
    assert only_food == pytest.approx(p_alt / p_ref, rel=1e-14)


def test_tables():
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ)
    same = micro.microsimulate(toy.survey(), ref, ref)
    ineq = micro.inequality(same)
    np.testing.assert_allclose(ineq["change"], 0.0, atol=1e-15)
    pov = micro.poverty(same, [3.0, 5.0])
    np.testing.assert_allclose(pov["change"], 0.0, atol=1e-15)
    alt = cge.solve_period(econ, dual(1.1))
    res = micro.microsimulate(toy.survey(), ref, alt)
    d = res.welfare()
    assert micro.poverty(res, 4.0).at[(4.0, 0), "alt"] == pytest.approx(
        micro.fgt(d["alt"], 4.0, 0, d["persons_alt"]), rel=1e-15)
    by = micro.poverty(res, 4.0, by_group=True)
    for col in ("ref", "alt"):
        assert by[(col, "hh_1")].iloc[0] + by[(col, "hh_2")].iloc[0] == pytest.approx(
            by[(col, "total")].iloc[0], rel=1e-14)
    # Per adult equivalent with the square-root scale.
    sq = res.welfare(0.5)
    np.testing.assert_allclose(sq["alt"], res.frame["real_alt"] / np.sqrt(res.frame["size"]))


def test_survey_misuse_is_refused():
    frame = toy.survey().frame
    sources = {"formal": "f_for", "informal": "f_inf", "capital": "f_cap"}
    with pytest.raises(ValueError, match="sources is empty"):
        micro.Survey(frame, sources={})
    with pytest.raises(KeyError, match="not in the survey"):
        micro.Survey(frame, sources={"wages": "f_for"})
    bad = frame.copy()
    bad.loc[0, "weight"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        micro.Survey(bad, sources=sources)
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ)
    other = frame.copy()
    other.loc[0, "group"] = "hh_9"
    with pytest.raises(ValueError, match="not household accounts"):
        micro.microsimulate(micro.Survey(other, sources=sources), ref, ref)
    with pytest.raises(ValueError, match="unknown income sources"):
        micro.microsimulate(micro.Survey(frame, sources={"formal": "f_lab"}), ref, ref)
    with pytest.raises(ValueError, match="do not name commodities"):
        micro.microsimulate(micro.Survey(frame, sources=sources, budget={"food": "c_x"}),
                            ref, ref)
    with pytest.raises(ValueError, match="disagree"):
        micro.microsimulate(micro.Survey(frame, sources={"formal": "f_for"}), ref, ref)
    only_one = frame[frame["group"] == "hh_1"].reset_index(drop=True)
    with pytest.raises(ValueError, match="no survey household"):
        micro.microsimulate(micro.Survey(only_one, sources=sources), ref, ref)
    with pytest.raises(ValueError, match="price_index"):
        micro.microsimulate(toy.survey(), ref, ref, price_index="paasche")


# -- reweighting ---------------------------------------------------------------

def test_raking_on_cells_is_the_ratio_in_each_cell():
    # Constraints that partition the sample: each cell's weights scale by
    # target / total, a closed form.
    w = np.array([1.0, 2.0, 3.0, 1.5, 0.5, 2.5])
    cells = np.array([0, 0, 1, 1, 2, 2])
    X = np.eye(3)[cells]
    targets = np.array([4.0, 3.0, 9.0])
    new, _ = micro._rake(w, X, targets)
    totals = X.T @ w
    np.testing.assert_allclose(new, w * (targets / totals)[cells], rtol=1e-12)


def test_raking_without_a_change_keeps_the_weights():
    w = RNG.uniform(0.5, 2.0, size=10)
    X = np.column_stack([np.ones(10), RNG.uniform(size=10)])
    new, iters = micro._rake(w, X, X.T @ w)
    np.testing.assert_allclose(new, w, rtol=1e-14)
    assert iters == 0


def test_raking_refuses_a_target_out_of_reach():
    w = np.ones(3)
    X = np.column_stack([np.ones(3), np.array([1.0, 1.0, 1.0])])
    with pytest.raises(ValueError, match="cannot be calibrated"):
        micro._rake(w, X, np.array([3.0, 5.0]), names=["households", "f_for"], group="hh_1")


def segmented_survey():
    s = toy.survey()
    frame = s.frame.copy()
    # Head counts: who in each household works formally, informally, or not.
    frame["n_formal"] = (frame["formal"] > 0).astype(float) + (frame["formal"] > 9)
    frame["n_informal"] = (frame["informal"] > 0).astype(float)
    frame["n_unemployed"] = [0.0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    # A household in each account whose members are all out of work: without
    # one, no reweighting can raise unemployment while employment falls in
    # both segments, as Harris–Todaro asks.
    idle = pd.DataFrame([("hh_1", 1.0, 2, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 2.0),
                         ("hh_2", 1.5, 3, 0.0, 0.0, 0.2, 1.0, 0.5, 0.0, 0.0, 2.0)],
                        columns=frame.columns)
    frame = pd.concat([frame, idle], ignore_index=True)
    return micro.Survey(frame, sources=s.sources, budget=s.budget,
                        workers={"n_formal": "f_for", "n_informal": "f_inf",
                                 "n_unemployed": "unemployed"})


@pytest.mark.parametrize("market", ["dual", "harris_todaro"])
def test_reweighting_moves_employment_as_the_model_does(market):
    _, econ = toy.build_with_segmented_labour()
    if market == "dual":
        closure_ref, closure_alt = dual(1.0), dual(1.1)
    else:
        def mk(wage):
            return cge.Closure(labour_markets={"labour": cge.LabourMarket(
                rigid={"f_for": wage}, informal="f_inf", unemployment="harris_todaro",
                unemployment_rate=0.1)})
        # The queue answers strongly to the rigid wage (+25% for a wage 2% higher),
        # and a survey of a few households spans only so much of it.
        closure_ref, closure_alt = mk(1.0), mk(1.02)
    ref, alt = cge.solve_period(econ, closure_ref), cge.solve_period(econ, closure_alt)
    res = micro.microsimulate(segmented_survey(), ref, alt)
    f = res.frame
    w, w_alt = f["weight"].to_numpy(), f["weight_alt"].to_numpy()
    mk_ref, mk_alt = ref.labour_markets["labour"], alt.labour_markets["labour"]
    for hi, h in enumerate(econ.hhs):
        m = (f["group"] == h).to_numpy()
        # Households are as many as before.
        assert w_alt[m].sum() == pytest.approx(w[m].sum(), rel=1e-12)
        for k, src in enumerate(["f_for", "f_inf"]):
            ratio = mk_alt["household_employment"][hi, k] / mk_ref["household_employment"][hi, k]
            assert w_alt[m] @ res.income_ref.loc[m, src] == pytest.approx(
                ratio * (w[m] @ res.income_ref.loc[m, src]), rel=1e-10)
        # Capital stays with the households that had it.
        assert w_alt[m] @ res.income_ref.loc[m, "f_cap"] == pytest.approx(
            w[m] @ res.income_ref.loc[m, "f_cap"], rel=1e-10)
        if market == "harris_todaro":
            u_ratio = mk_alt["household_unemployed"][hi] / mk_ref["household_unemployed"][hi]
            row = res.workers.set_index(["group", "column"]).loc[(h, "n_unemployed")]
            assert row["alt"] == pytest.approx(u_ratio * row["ref"], rel=1e-10)
    # Micro totals are still the model's, source by source.
    A_alt = micro.account_sources(alt)
    for h, src in itertools.product(econ.hhs, ["f_cap", "f_for", "f_inf"]):
        m = (f["group"] == h).to_numpy()
        assert w_alt[m] @ res.income_alt.loc[m, src] == pytest.approx(A_alt.at[h, src],
                                                                      rel=1e-10)
    cal = res.calibration
    np.testing.assert_allclose(cal["reached"], cal["target"], rtol=1e-10)
    # A higher formal wage hires fewer formal workers in the survey too.
    formal = res.workers.set_index("column").loc["n_formal"]
    assert formal["alt"].sum() < formal["ref"].sum()


def test_reweighting_without_a_shock_changes_nothing():
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ, dual(1.0))
    res = micro.microsimulate(segmented_survey(), ref, ref)
    np.testing.assert_allclose(res.frame["weight_alt"], res.frame["weight"], rtol=1e-14)
    assert (res.calibration["iterations"] == 0).all()


def test_the_tables_use_each_periods_weights():
    _, econ = toy.build_with_segmented_labour()
    ref, alt = cge.solve_period(econ, dual(1.0)), cge.solve_period(econ, dual(1.1))
    res = micro.microsimulate(segmented_survey(), ref, alt)
    d = res.welfare()
    assert micro.inequality(res).at["gini", "alt"] == pytest.approx(
        micro.gini(d["alt"], d["persons_alt"]), rel=1e-15)
    assert micro.inequality(res).at["gini", "ref"] == pytest.approx(
        micro.gini(d["ref"], d["persons_ref"]), rel=1e-15)
    plain = micro.microsimulate(segmented_survey(), ref, alt, reweight=False)
    np.testing.assert_allclose(plain.frame["weight_alt"], plain.frame["weight"])
    assert plain.calibration.empty


def test_a_segment_mapped_by_activity_is_refused_for_reweighting():
    _, econ = toy.build_with_segmented_labour()
    ref = cge.solve_period(econ)
    s = toy.survey()
    by_activity = micro.Survey(s.frame, sources={"formal": "f_for:a_ind", "informal": "f_inf",
                                                 "capital": "f_cap"}, budget=s.budget)
    with pytest.raises(ValueError, match="mapped by activity"):
        micro.microsimulate(by_activity, ref, ref)
