"""Monopoly and fixed costs: Hosoe's moncge and irscge, against the transcription of the GAMS."""

import json
from pathlib import Path

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import hosoe

REF = json.loads((Path(__file__).parent / "data" / "hosoe_market.json").read_text())
CASES = {"base": None, "abolicao": cge.Scenario(tariff=np.zeros(2))}
BUILD = {"moncge": hosoe.build_monopoly, "irscge": hosoe.build_scale}


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", float(v)


@pytest.mark.parametrize("model", list(BUILD))
@pytest.mark.parametrize("case", list(CASES))
def test_every_variable_matches_the_transcription_of_the_gams_program(model, case):
    _, econ = BUILD[model]()
    period = cge.solve_period(econ, hosoe.CLOSURE, CASES[case])
    ours = dict(_flat(hosoe.variables(econ, period)))
    ref = dict(_flat(REF[model][case]))
    common = sorted(set(ours) & set(ref))
    assert len(common) >= 45
    for k in common:
        assert ours[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-9), k


def test_the_monopoly_rent_is_the_lerner_markup_on_home_sales():
    _, econ = hosoe.build_monopoly()
    np.testing.assert_allclose(econ.monopoly["rent0"], [70.0 / 3.0, 72.0 / 3.0])
    np.testing.assert_allclose(econ.monopoly["markup"], 1.5)
    r = cge.solve_period(econ, hosoe.CLOSURE, cge.Scenario(tariff=np.zeros(2)))
    for c, m in r.monopoly.items():
        # The markup's share of what buyers pay: (1 - 1/1.5) of price times home sales.
        p = r.p_domestic[econ.commodity(c)]
        assert m["rent"] == pytest.approx(p * m["home_sales"] / 3.0, rel=1e-10)


@pytest.mark.parametrize("model", list(BUILD))
def test_accounts_close_and_the_welfare_identity_holds(model):
    _, econ = BUILD[model]()
    sc = cge.Scenario(tariff=np.zeros(2))
    acc = cge.national_accounts(cge.solve_period(econ, hosoe.CLOSURE, sc))
    assert abs(acc["gap"]) < 1e-9
    w = cge.welfare_decomposition(econ, cge.Scenario(), sc, closure=hosoe.CLOSURE)
    assert abs(w.residual) < 1e-8
    term = "allocative_monopoly" if model == "moncge" else "scale_economies"
    assert abs(w[term]) > 1e-4


def test_fixed_costs_give_increasing_returns():
    _, econ = hosoe.build_scale()
    base = cge.solve_period(econ, hosoe.CLOSURE)
    more = cge.solve_period(econ, hosoe.CLOSURE, cge.Scenario(tariff=np.zeros(2)))
    for a, f in more.fixed_costs.items():
        assert f["average"] == pytest.approx(f["fixed_cost"] / f["output"], rel=1e-10)
    rises = [a for a in econ.acts if more.fixed_costs[a]["output"] > base.fixed_costs[a]["output"]]
    assert all(more.fixed_costs[a]["average"] < base.fixed_costs[a]["average"] for a in rises)


def test_market_structure_is_refused_where_it_would_not_hold():
    sam, _ = hosoe.build()
    with pytest.raises(ValueError, match="above one"):
        cge.calibrate(sam, hosoe.spec(sigma_armington=1.0, monopoly=["c_brd"]))
    with pytest.raises(KeyError, match="not a commodity"):
        cge.calibrate(sam, hosoe.spec(monopoly=["c_xyz"]))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        cge.calibrate(sam, hosoe.spec(fixed_costs={"a_brd": 1.5}))
    _, econ = hosoe.build_monopoly()
    with pytest.raises(ValueError, match="monopoly"):
        cge.to_jump(econ)
