"""Direct taxes, transfers, foreign ownership, FDI, per-year economies and several capitals."""

import dataclasses

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy

CLOSURES = [
    cge.Closure(),
    cge.Closure(foreign="fixed_rate"),
    cge.Closure(investment="fixed_real"),
    cge.Closure(government="nominal_fixed"),
    cge.Closure(investment="fixed_real", foreign="fixed_rate"),
]


def institutions():
    return toy.build_with_institutions()[1]


# ---------------------------------------------------------------------------
# Calibration of the institutional flows
# ---------------------------------------------------------------------------

def test_institutional_flows_are_calibrated_from_the_sam():
    econ = institutions()
    h1, h2 = econ.household("hh_1"), econ.household("hh_2")
    kap = econ.factor("f_cap")
    # hh_1: receipts 105, of which 1 is sent abroad, so gross income is 104
    assert econ.hh_dtax[h1] == pytest.approx(12 / 104)
    assert econ.hh_savrate[h1] == pytest.approx(7 / 92)          # share of DISPOSABLE income
    assert econ.hh_gov_transfer[h2] == pytest.approx(8.0)
    assert econ.hh_remit_net[h2] == pytest.approx(3.0)
    assert econ.hh_remit_net[h1] == pytest.approx(-1.0)
    assert econ.fac_gov_share[kap] == pytest.approx(4 / 60)
    assert econ.fac_row_share[kap] == pytest.approx(10 / 60)
    assert econ.foreign_sav == pytest.approx(18.0)                # the current account
    assert econ.trade_balance0 == pytest.approx(10.0)             # trade alone
    assert econ.diagnostics["unmodelled_flows"] == []


def test_an_unmodelled_flow_is_named():
    """A gov <- row grant is not in the model; the calibration says so."""
    sam, _ = toy.build_with_institutions()
    df = sam.frame.copy()
    df.loc[len(df)] = ("gov", "row", 2.0)
    df.loc[len(df)] = ("s_i", "gov", 2.0)
    df.loc[len(df)] = ("row", "s_i", 2.0)          # keeps every account balanced
    econ = cge.calibrate(cge.SAM.from_long(df), toy.spec())
    names = {(r, c) for r, c, _ in econ.diagnostics["unmodelled_flows"]}
    assert ("gov", "row") in names


def test_institutions_reproduce_the_base_year():
    rep = cge.base_year_reproduction(institutions())
    assert rep["ok"], rep["checks"]
    assert abs(rep["walras"]) < 1e-10


def test_institutions_are_homogeneous():
    rep = cge.homogeneity_test(institutions(), factor=3.0)
    assert rep["ok"], rep["checks"]


@pytest.mark.parametrize("closure", CLOSURES, ids=lambda c: c.describe()[12:])
def test_institutions_keep_walras_under_a_shock_in_every_closure(closure):
    econ = institutions()
    res = cge.solve_period(econ, closure, cge.Scenario(product_tax=econ.ptax * 1.2))
    assert res.converged
    assert abs(res.walras) < 1e-9


def test_a_higher_income_tax_funds_investment_and_cuts_consumption():
    econ = institutions()
    base = cge.solve_period(econ)
    taxed = cge.solve_period(dataclasses.replace(econ, hh_dtax=econ.hh_dtax * 1.5))
    assert abs(taxed.walras) < 1e-9
    assert taxed.hh_direct_tax.sum() > base.hh_direct_tax.sum()
    assert taxed.real["hh_real"] < base.real["hh_real"]
    assert taxed.real["inv_real"] > base.real["inv_real"]


def test_foreign_ownership_sends_capital_income_abroad():
    econ = institutions()
    kap = econ.factor("f_cap")
    base = cge.solve_period(econ)
    share = econ.fac_row_share.copy()
    share[kap] = 0.4
    res = cge.solve_period(econ, scenario=cge.Scenario(foreign_factor_share=share))
    assert abs(res.walras) < 1e-9
    assert res.factor_income_abroad > base.factor_income_abroad
    assert res.hh_income.sum() < base.hh_income.sum()


def test_fdi_raises_foreign_saving_and_investment():
    econ = institutions()
    base = cge.solve_period(econ)
    res = cge.solve_period(econ, scenario=cge.Scenario(fdi={"f_cap": 5.0}))
    assert abs(res.walras) < 1e-9
    assert res.foreign_saving == pytest.approx(base.foreign_saving + 5.0)
    assert res.real["inv_real"] > base.real["inv_real"]


def test_fdi_into_an_unknown_factor_is_refused():
    with pytest.raises(ValueError):
        cge.solve_period(institutions(), scenario=cge.Scenario(fdi={"f_nope": 1.0}))


def test_to_jump_exports_the_institutions(tmp_path):
    econ = institutions()
    kap = econ.factor("f_cap")
    share = econ.fac_row_share.copy()
    share[kap] = 0.4
    nu = 2.0
    dados = cge.to_jump(econ, cge.Closure(numeraire=nu),
                        cge.Scenario(foreign_factor_share=share, fdi={"f_cap": 5.0}),
                        path=tmp_path / "m.jl")
    assert np.allclose(dados["hh_dtax"], econ.hh_dtax)
    assert np.allclose(dados["gov_transfer"], econ.hh_gov_transfer * nu)   # numeraire units
    assert np.allclose(dados["remit_net"], econ.hh_remit_net * nu)
    assert np.allclose(dados["fac_row_share"], share)                      # the scenario's
    assert dados["foreign_sav"] == pytest.approx((econ.foreign_sav + 5.0) * nu)
    assert "current == fsav" in (tmp_path / "m.jl").read_text()


# ---------------------------------------------------------------------------
# Dynamics: per-year economies, foreign-owned capital, several capitals
# ---------------------------------------------------------------------------

YEARS = range(2020, 2031)
GROWTH = cge.Growth(labour=0.01, tfp=0.01)


def test_economy_hook_changes_the_path_and_keeps_walras():
    _, econ = toy.build()
    ref = cge.run_path(econ, years=YEARS, growth=GROWTH)
    alt = cge.run_path(econ, years=YEARS, growth=GROWTH,
                       economy=lambda y, t: dataclasses.replace(
                          econ, foreign_sav=econ.foreign_sav + 0.5 * t))
    assert alt.converged and alt.worst_walras < 1e-9
    assert alt.capital_stock[-1] > ref.capital_stock[-1]


def test_economy_hook_must_keep_the_accounts():
    _, econ = toy.build()
    _, other = toy.build_with_own_production()
    with pytest.raises(ValueError, match="different accounts"):
        cge.run_path(econ, years=YEARS, economy=lambda y, t: other)


def test_fdi_capital_is_foreign_owned_and_its_returns_leave():
    """The same inflow, owned abroad versus handed to residents."""
    econ = institutions()
    fdi = lambda y, t: cge.Scenario(fdi={"f_cap": 3.0})
    owned = cge.run_path(econ, years=YEARS, growth=GROWTH, scenario=fdi)
    gift = cge.run_path(econ, years=YEARS, growth=GROWTH,
                        economy=lambda y, t: dataclasses.replace(
                           econ, foreign_sav=econ.foreign_sav + 3.0))
    assert owned.converged and owned.worst_walras < 1e-9
    assert gift.worst_walras < 1e-9
    # the same inflow builds the same capital in the first year, but when it is
    # owned abroad its returns leave, residents earn and save less, and the
    # stock falls behind; the foreign part ends above the path with no FDI
    assert owned.capital_stock[1] == pytest.approx(gift.capital_stock[1], rel=1e-12)
    assert owned.capital_stock[-1] < gift.capital_stock[-1]
    none = cge.run_path(econ, years=YEARS, growth=GROWTH)
    assert owned.foreign_capital[-1] > none.foreign_capital[-1]
    assert owned[2030].factor_income_abroad > gift[2030].factor_income_abroad
    assert owned[2030].hh_income.sum() < gift[2030].hh_income.sum()


def test_the_sam_foreign_share_is_kept_with_no_fdi():
    econ = institutions()
    path = cge.run_path(econ, years=YEARS, growth=GROWTH)
    share = path.foreign_capital / path.capital_stock
    np.testing.assert_allclose(share[0], econ.fac_row_share[econ.factor("f_cap")], rtol=1e-12)
    assert path.worst_walras < 1e-9


def specific():
    return toy.build_with_specific_capital()[1]


def test_several_capitals_keep_their_composition_with_no_shock():
    econ = specific()
    g = cge.Growth(labour=0.01, tfp=0.01, capital_factor=("f_cap", "f_kind"))
    path = cge.run_path(econ, years=YEARS, growth=g)
    assert path.converged and path.worst_walras < 1e-9
    assert path.capital_stock.shape == (len(YEARS), 2)
    ratio = path.capital_stock[:, 1] / path.capital_stock[:, 0]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


def test_investment_follows_rents_when_allocation_responds():
    """A demand boom for industry raises the rent on its specific capital."""
    econ = specific()
    boom = lambda y, t: cge.Scenario(world_price=np.array([1.0, 1.3]))
    prop = cge.run_path(econ, years=YEARS, scenario=boom,
                        growth=cge.Growth(capital_factor=("f_cap", "f_kind")))
    resp = cge.run_path(econ, years=YEARS, scenario=boom,
                        growth=cge.Growth(capital_factor=("f_cap", "f_kind"), allocation=2.0))
    assert resp.worst_walras < 1e-9
    k = econ.factor("f_kind")
    c = econ.factor("f_cap")
    first = resp[2020]
    tilt = first.factor_prices[k] / first.factor_prices[c]
    share_resp = resp.capital_stock[-1, 1] / resp.capital_stock[-1].sum()
    share_prop = prop.capital_stock[-1, 1] / prop.capital_stock[-1].sum()
    assert (share_resp > share_prop) == (tilt > 1.0)


def test_fdi_into_a_specific_capital():
    econ = specific()
    g = cge.Growth(capital_factor=("f_cap", "f_kind"))
    path = cge.run_path(econ, years=YEARS, growth=g,
                        scenario=lambda y, t: cge.Scenario(fdi={"f_kind": 2.0}))
    assert path.worst_walras < 1e-9
    assert path.foreign_capital[-1, 1] > 0.0
    assert path.foreign_capital[-1, 0] == 0.0


def test_depreciation_by_factor():
    econ = specific()
    g = cge.Growth(capital_factor=("f_cap", "f_kind"), depreciation={"f_cap": 0.05})
    with pytest.raises(KeyError, match="f_kind"):
        cge.run_path(econ, years=YEARS, growth=g)
    g = cge.Growth(capital_factor=("f_cap", "f_kind"),
                   depreciation={"f_cap": 0.05, "f_kind": 0.10})
    assert cge.run_path(econ, years=YEARS, growth=g).worst_walras < 1e-9


def test_fdi_must_target_an_accumulated_capital():
    with pytest.raises(ValueError, match="does not accumulate"):
        cge.run_path(institutions(), years=YEARS,
                     scenario=lambda y, t: cge.Scenario(fdi={"f_lab": 1.0}))


def test_direct_tax_reduces_welfare_income():
    econ = institutions()
    path = cge.run_path(econ, years=range(2020, 2023))
    hh = path.households()
    assert (hh.direct_tax > 0).any()
    ev = cge.equivalent_variation(path, path)
    np.testing.assert_allclose(ev.ev, 0.0, atol=1e-9)
