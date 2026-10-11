"""CET exports and the external closure's single source of truth."""

import dataclasses

import numpy as np
import pytest

import cgelab as ck
from cgelab.examples import toy

VARIANTS = [("plain", toy.build), ("margins", toy.build_with_margins),
            ("own_account", toy.build_with_own_production)]
CLOSURES = [
    ck.Closure(),
    ck.Closure(foreign="fixed_rate"),
    ck.Closure(investment="fixed_real"),
    ck.Closure(government="nominal_fixed"),
    ck.Closure(investment="fixed_real", foreign="fixed_rate"),
]


def cet_economy(build=toy.build, sigma=2.0):
    return build(model_spec=toy.spec(export_rule="cet", sigma_cet=sigma))[1]


# ---------------------------------------------------------------------------
# CET
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,build", VARIANTS)
def test_cet_reproduces_the_base_year(label, build):
    """At base-year prices the transformation frontier returns the SAM's split."""
    rep = ck.base_year_reproduction(cet_economy(build))
    assert rep["ok"], f"{label}: worst {rep['worst']:.2e}, checks {rep['checks']}"


@pytest.mark.parametrize("label,build", VARIANTS)
def test_cet_is_homogeneous_of_degree_zero(label, build):
    rep = ck.homogeneity_test(cet_economy(build), factor=3.0)
    assert rep["ok"], f"{label}: worst {rep['worst']:.2e}, checks {rep['checks']}"


@pytest.mark.parametrize("closure", CLOSURES, ids=lambda c: c.describe()[12:])
def test_cet_keeps_walras_under_tax_and_export_price_shocks(closure):
    econ = cet_economy()
    sc = ck.Scenario(product_tax=econ.ptax * 1.2, export_price=np.array([1.15, 0.9]))
    res = ck.solve_period(econ, closure, sc)
    assert res.converged
    assert abs(res.walras) < 1e-9


def test_a_higher_export_price_shifts_output_abroad():
    econ = cet_economy()
    j = econ.commodity("c_ind")
    base = ck.solve_period(econ)
    pe = np.ones(econ.n_com)
    pe[j] = 1.2
    res = ck.solve_period(econ, scenario=ck.Scenario(export_price=pe))
    assert res.exports[j] / res.Q[j] > base.exports[j] / base.Q[j]
    assert res.pe[j] > res.pd[j]          # exports now fetch more than home sales


def test_the_export_response_grows_with_the_cet_elasticity():
    shares = []
    for sigma in (0.0, 1.0, 4.0):
        econ = cet_economy(sigma=sigma)
        j = econ.commodity("c_ind")
        pe = np.ones(econ.n_com)
        pe[j] = 1.2
        res = ck.solve_period(econ, scenario=ck.Scenario(export_price=pe))
        shares.append(res.exports[j] / res.Q[j])
    assert shares[0] == pytest.approx(cet_economy().exp_share[1], rel=1e-9)  # sigma 0: no response
    assert shares[0] < shares[1] < shares[2]


def test_cet_value_identity_holds():
    """Output revenue equals revenue from home sales plus revenue from exports."""
    econ = cet_economy()
    res = ck.solve_period(econ, scenario=ck.Scenario(export_price=np.array([1.1, 0.95])))
    p_dom = ck.cet_domestic_price(econ.exp_share, res.pd, res.pe, econ.sigma_cet)
    _, dom_frac, exp_frac = ck.cet(econ.exp_share, p_dom, res.pe, econ.sigma_cet)
    np.testing.assert_allclose(res.pd * res.Q,
                               p_dom * dom_frac * res.Q + res.pe * exp_frac * res.Q, rtol=1e-12)


def test_export_price_is_refused_under_fixed_shares():
    """A field the model would ignore is an error, not a no-op."""
    _, econ = toy.build()
    with pytest.raises(ValueError, match="export_price"):
        ck.solve_period(econ, scenario=ck.Scenario(export_price=np.array([1.1, 1.0])))


def test_cet_path_keeps_walras_at_every_date():
    econ = cet_economy()
    path = ck.run_path(econ, years=range(2020, 2031),
                       scenario=lambda y, t: ck.Scenario(
                           export_price=np.array([1.0 + 0.01 * t, 1.0])),
                       growth=ck.Growth(labour=0.01, tfp=0.01))
    assert path.converged
    assert path.worst_walras < 1e-9


def test_to_jump_exports_the_cet_block(tmp_path):
    econ = cet_economy(sigma=3.0)
    pe = np.array([1.1, 0.9])
    dados = ck.to_jump(econ, scenario=ck.Scenario(export_price=pe), path=tmp_path / "m.jl")
    assert dados["cet"] is True
    assert np.allclose(dados["export_price"], pe)
    assert np.allclose(dados["sigma_cet"], 3.0)
    assert "pdom" in (tmp_path / "m.jl").read_text()


# ---------------------------------------------------------------------------
# External closure: foreign saving is the current-account target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [toy.build, lambda: toy.build(
    model_spec=toy.spec(export_rule="cet"))], ids=["fixed_shares", "cet"])
def test_moving_foreign_saving_alone_keeps_walras(build):
    """`foreign_sav` is the one lever. It used to need `trade_balance0` moved in step,
    and moving it alone left a Walras residual that grew along a path."""
    _, econ = build()
    more = dataclasses.replace(econ, foreign_sav=econ.foreign_sav + 5.0)
    res = ck.solve_period(more)
    assert res.converged
    assert abs(res.walras) < 1e-9
    trade = float((res.exchange_rate * res.imports).sum() - (res.pe * res.exports).sum())
    assert trade == pytest.approx(more.foreign_sav, rel=1e-9)
    assert res.investment_quantity.sum() > ck.solve_period(econ).investment_quantity.sum()
