"""Disaggregating a SAM with household surveys.

Splitting and summing back must be the identity. Two economic equivalences
check that the split changes nothing it should not: households split into
groups with identical proportions have, with Cobb–Douglas demand, the
aggregate's equilibrium; and labour split into formal and informal, as a group
of segments with one wage, has the single factor's equilibrium, because a CES
with two inputs at the same price is the CES with their sum.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab import ibge
from cgelab.examples import toy
from pathlib import Path

SHOCK = cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.0, 1.2]))
DATA = Path(__file__).resolve().parent / "data" / "ibge"


def matrix(sam, accounts):
    return sam.matrix(accounts)


def same_equilibrium(a, b, tol=1e-10):
    for name in ("X", "Q", "pq", "imports", "exports"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=tol, atol=tol,
                                   err_msg=name)


# -- households ---------------------------------------------------------------

def hh_1_survey():
    s = toy.survey()
    f = s.frame[s.frame["group"] == "hh_1"].copy()
    f["group"] = ["hh_1a", "hh_1a", "hh_1a", "hh_1b", "hh_1b", "hh_1b"]
    return cge.Survey(f, sources=s.sources, budget=s.budget)


def test_splitting_households_and_summing_them_back():
    sam, _ = toy.build_with_segmented_labour()
    survey = hh_1_survey()
    split = cge.split_households(sam, survey, account="hh_1")
    split.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    M, S = sam.matrix(), split.matrix()
    groups = ["hh_1a", "hh_1b"]
    others = [a for a in M.index if a != "hh_1"]
    # Summing the groups gives back the account, row and column.
    np.testing.assert_allclose(S.loc[groups, others].sum(axis=0), M.loc["hh_1", others],
                               atol=1e-12)
    np.testing.assert_allclose(S.loc[others, groups].sum(axis=1), M.loc[others, "hh_1"],
                               atol=1e-12)
    np.testing.assert_allclose(S.loc[others, others], M.loc[others, others], atol=0)
    # Each cell is shared by the group's share of the weighted survey total.
    f = survey.frame
    wy = f["weight"] * f["formal"]
    share_a = wy[f["group"] == "hh_1a"].sum() / wy.sum()
    assert S.at["hh_1a", "f_for"] == pytest.approx(60.0 * share_a, rel=1e-14)
    wf = f["weight"] * f["food"]
    assert S.at["c_agr", "hh_1b"] == pytest.approx(
        30.0 * wf[f["group"] == "hh_1b"].sum() / wf.sum(), rel=1e-14)
    # Saving is the residual, and the provenance says so.
    p = split.provenance.set_index("note")
    assert p.at["saving of hh_1a", "source"] == "residual" or \
        p.at["saving of hh_1a (negative)", "source"] == "residual"
    pop = cge.group_population(survey)
    assert pop["hh_1a"] == pytest.approx(2.0 * 4 + 1.5 * 3 + 1.0 * 2)


def test_households_with_identical_proportions_have_the_aggregate_equilibrium():
    sam, econ = toy.build()
    cells = {"lab": 70.0, "cap": 45.0, "agr": 30.0, "ind": 55.0}      # hh_1's row and column
    scale = [1.0, 2.5, 0.5, 4.0]
    frame = pd.DataFrame({"group": ["hh_1a", "hh_1b", "hh_1c", "hh_1c"], "weight": 1.0,
                          "size": 1.0, **{k: [v * s for s in scale] for k, v in cells.items()}})
    survey = cge.Survey(frame, sources={"lab": "f_lab", "cap": "f_cap"},
                        budget={"agr": "c_agr", "ind": "c_ind"})
    split = cge.split_households(sam, survey, account="hh_1")
    econ_split = cge.calibrate(split, toy.spec())
    rep = cge.base_year_reproduction(econ_split)
    assert rep["ok"], rep["checks"]
    a = cge.solve_period(econ, scenario=SHOCK)
    b = cge.solve_period(econ_split, scenario=SHOCK)
    same_equilibrium(a, b)
    hh1 = [econ_split.household(h) for h in ("hh_1a", "hh_1b", "hh_1c")]
    assert b.hh_income[hh1].sum() == pytest.approx(a.hh_income[econ.household("hh_1")],
                                                    rel=1e-10)


def test_a_cell_the_survey_cannot_share():
    sam, _ = toy.build_with_institutions()            # hh_1 pays a direct tax and sends money abroad
    s = toy.survey()
    f = s.frame[s.frame["group"] == "hh_1"].copy()
    f["group"] = ["hh_1a"] * 3 + ["hh_1b"] * 3
    survey = cge.Survey(f, sources={"formal": "f_lab", "capital": "f_cap"},
                        budget={"food": "c_agr", "goods": "c_ind"})
    with pytest.raises(ValueError, match="gov"):
        cge.split_households(sam, survey, account="hh_1")
    f["tax"] = [1.0, 2.0, 3.0, 0.5, 0.5, 4.0]
    survey = cge.Survey(f, sources={"formal": "f_lab", "capital": "f_cap"},
                        budget={"food": "c_agr", "goods": "c_ind"})
    split = cge.split_households(sam, survey, account="hh_1", outlays={"gov": "tax"},
                                 fallback={"row": "population"})
    split.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    p = split.provenance.set_index("note")
    assert "spending on row: by population" in p.index
    persons = f["weight"] * f["size"]
    assert split.matrix().at["row", "hh_1a"] == pytest.approx(
        1.0 * persons[f["group"] == "hh_1a"].sum() / persons.sum(), rel=1e-14)


def test_household_misuse():
    sam, _ = toy.build_with_segmented_labour()
    s = toy.survey()                                   # its groups are hh_1 and hh_2
    with pytest.raises(ValueError, match="already accounts"):
        cge.split_households(sam, s, account="hh_1")
    with pytest.raises(KeyError, match="not an account"):
        cge.split_households(sam, hh_1_survey(), account="hh_9")


# -- factors ------------------------------------------------------------------

def formal_informal():
    return pd.DataFrame([[0.3, 0.8], [0.7, 0.2]], index=["f_for", "f_inf"],
                        columns=["a_agr", "a_ind"])


def test_splitting_a_factor_and_summing_it_back():
    sam, _ = toy.build()
    split = cge.split_factor(sam, "f_lab", formal_informal())
    split.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    M, S = sam.matrix(), split.matrix()
    new = ["f_for", "f_inf"]
    others = [a for a in M.index if a != "f_lab"]
    np.testing.assert_allclose(S.loc[new, others].sum(axis=0), M.loc["f_lab", others],
                               atol=1e-12)
    np.testing.assert_allclose(S.loc[others, new].sum(axis=1), M.loc[others, "f_lab"],
                               atol=1e-12)
    assert S.at["f_for", "a_agr"] == pytest.approx(40.0 * 0.3)
    assert S.at["f_inf", "a_ind"] == pytest.approx(70.0 * 0.2)
    # Without owners every household keeps its share of each part.
    assert S.at["hh_1", "f_for"] / S.loc[["hh_1", "hh_2"], "f_for"].sum() == pytest.approx(
        70.0 / 110.0)


def test_labour_split_into_segments_with_one_wage_is_the_single_factor():
    sam, _ = toy.build()
    split = cge.split_factor(sam, "f_lab", formal_informal())
    one = cge.calibrate(sam, toy.spec(numeraire_factor="f_cap"))
    two = cge.calibrate(split, toy.spec(factors=("f_cap", "f_for", "f_inf"),
                                        numeraire_factor="f_cap",
                                        labour_segments={"labour": ["f_for", "f_inf"]}))
    rep = cge.base_year_reproduction(two)
    assert rep["ok"], rep["checks"]
    a, b = cge.solve_period(one, scenario=SHOCK), cge.solve_period(two, scenario=SHOCK)
    same_equilibrium(a, b)
    np.testing.assert_allclose(b.hh_income, a.hh_income, rtol=1e-10)
    np.testing.assert_allclose(b.factor_prices[1:], a.factor_prices[0], rtol=1e-10)


def test_the_pnad_continua_splits_labour():
    survey, persons = ibge.pnadc(
        DATA / "PNADC_teste.txt", DATA / "pnadc_input.txt",
        segments={"formal": "f_for", "informal": "f_inf"},
        group=lambda h: np.where(h["household"].str.startswith("11"), "hh_1", "hh_2"),
        activities={"01": "a_agr", "2": "a_ind", "4": "a_ind", "97": "a_ind"})
    by_activity, owners = cge.segment_shares(survey, persons, segments=["f_for", "f_inf"])
    wA, wB = 1234.567890123, 987.25
    # Worked out from the fixture: who earns what, where, in which household.
    assert by_activity.at["f_for", "a_ind"] == pytest.approx(wA * 3000 + wB * 5000)
    assert by_activity.at["f_for", "a_agr"] == 0.0
    assert by_activity.at["f_inf", "a_agr"] == pytest.approx(wA * 1200 + wB * 0.0)
    assert by_activity.at["f_inf", "a_ind"] == pytest.approx(wB * 900)
    assert owners.at["hh_1", "f_for"] == pytest.approx(wA * 3000)
    assert owners.at["hh_2", "f_inf"] == pytest.approx(wB * 900)
    sam, _ = toy.build()
    split = cge.split_factor(sam, "f_lab", by_activity, owners=owners, source="PNADC fixture")
    split.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    S, M = split.matrix(), sam.matrix()
    # Agriculture pays only informal labour, as the fixture's farm workers are.
    assert S.at["f_for", "a_agr"] == 0.0 and S.at["f_inf", "a_agr"] == pytest.approx(40.0)
    # Households receive each part by their earnings in it, and save what is left.
    inc_for = S.loc["f_for"].sum()
    assert S.at["hh_1", "f_for"] == pytest.approx(inc_for * 3000 * wA
                                                  / (3000 * wA + 5000 * wB))
    assert S.loc["s_i", ["hh_1", "hh_2"]].sum() == pytest.approx(
        M.loc["s_i", ["hh_1", "hh_2"]].sum())
    p = split.provenance.set_index("note")
    assert p.at["f_lab split by activity", "source"] == "PNADC fixture"
    assert p.at["saving of hh_1, recomputed", "source"] == "residual"
    econ = cge.calibrate(split, toy.spec(factors=("f_cap", "f_for", "f_inf"),
                                         numeraire_factor="f_cap",
                                         labour_segments={"labour": ["f_for", "f_inf"]}))
    rep = cge.base_year_reproduction(econ)
    assert rep["ok"], rep["checks"]


def test_factor_misuse():
    sam, _ = toy.build()
    with pytest.raises(KeyError, match="not an account"):
        cge.split_factor(sam, "f_land", formal_informal())
    with pytest.raises(ValueError, match="a_ind"):
        cge.split_factor(sam, "f_lab", formal_informal()[["a_agr"]])
    with pytest.raises(ValueError, match="already accounts"):
        cge.split_factor(sam, "f_lab", formal_informal().rename(index={"f_for": "f_cap"}))
    owners = pd.DataFrame({"f_for": [1.0], "f_inf": [1.0]}, index=["gov"])
    with pytest.raises(ValueError, match="receive no"):
        cge.split_factor(sam, "f_lab", formal_informal(), owners=owners)
