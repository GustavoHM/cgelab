"""Two countries trading with each other: Hosoe's twocge, against the transcription of the GAMS."""

import json
from pathlib import Path

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import hosoe

REF = json.loads((Path(__file__).parent / "data" / "hosoe_twocge.json").read_text())
CASES = {"base": None, "abolicao": cge.Scenario(tariff=np.zeros(2))}
(_, JPN), (_, USA) = hosoe.build_two_countries()


def _solve(sc=None, **kw):
    return cge.solve_two_countries(JPN, USA, closure_a=hosoe.CLOSURE, closure_b=hosoe.CLOSURE,
                                   scenario_a=sc, scenario_b=sc, names=("JPN", "USA"), **kw)


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", float(v)


@pytest.mark.parametrize("case", list(CASES))
def test_every_variable_matches_the_transcription_of_twocge(case):
    r = _solve(CASES[case])
    for country, econ in (("JPN", JPN), ("USA", USA)):
        ours = dict(_flat(hosoe.variables(econ, r.periods[country])))
        ref = dict(_flat(REF[case][country]))
        common = sorted(set(ours) & set(ref))
        assert len(common) >= 45
        for k in common:
            assert ours[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-9), (country, k)


def test_trade_clears_and_both_balances_of_payments_hold():
    r = _solve(cge.Scenario(tariff=np.zeros(2)))
    np.testing.assert_allclose(r.trade["JPN exports"], r.trade["USA imports"], rtol=1e-10)
    np.testing.assert_allclose(r.trade["USA exports"], r.trade["JPN imports"], rtol=1e-10)
    for country in ("JPN", "USA"):
        p = r.periods[country]
        assert p.converged and abs(p.walras) < 1e-10
    assert r.periods["USA"].exchange_rate == pytest.approx(1.0, abs=1e-10)


def test_the_numeraire_country_matters_when_foreign_saving_is_fixed_in_dollars():
    """Foreign saving is fixed in dollars, so which exchange rate is held sets what it is worth.

    Without foreign saving the level of world prices would be neutral; with it, holding Japan's
    exchange rate instead of the United States' is another experiment, which is why twocge says
    which one it holds.
    """
    a = _solve(cge.Scenario(tariff=np.zeros(2)))
    b = _solve(cge.Scenario(tariff=np.zeros(2)), numeraire=0)
    assert b.periods["JPN"].exchange_rate == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(b.trade["JPN exports"], b.trade["USA imports"], rtol=1e-10)
    assert not np.allclose(b.periods["JPN"].hh_quantity, a.periods["JPN"].hh_quantity,
                           rtol=1e-6)


def test_two_countries_are_refused_where_they_would_not_hold():
    with pytest.raises(ValueError, match="world prices"):
        cge.solve_two_countries(JPN, USA, scenario_a=cge.Scenario(world_price=np.ones(2)))
    _, plain = hosoe.build()
    with pytest.raises(ValueError, match="are not the other's"):
        cge.solve_two_countries(JPN, plain)
    sam, _ = hosoe.build()
    fixed = cge.calibrate(sam, hosoe.spec(export_rule="fixed_shares"))
    with pytest.raises(ValueError, match="fixed export shares"):
        cge.solve_two_countries(fixed, USA)
