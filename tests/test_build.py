"""Building a SAM from supply and use tables.

The reference is the toy SAMs, written by hand cell by cell. Their blocks are
sliced out with ``SAM.block`` into supply and use tables and institutional
accounts, and :func:`sut_to_sam` must give back the original SAM, every cell,
savings included, though it computes those as residuals. A table that does not
add up must be refused where it fails.
"""

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import toy


def block(sam, rows, cols):
    rows = [r for r in rows if r in sam.accounts]
    cols = [c for c in cols if c in sam.accounts]
    return pd.DataFrame(sam.block(rows, cols), index=rows, columns=cols)


def tables(sam, *, margins=False, trade_taxes=False, own=False, institutions=False):
    """Supply and use tables and institutional accounts, sliced from a SAM."""
    acc = sam.accounts
    acts = [a for a in acc if a.startswith("a_")]
    coms = [c for c in acc if c.startswith("c_")]
    hhs = [h for h in acc if h.startswith("hh_")]
    facs = [f for f in acc if f.startswith("f_")]
    final = block(sam, coms, hhs + ["gov", "s_i", "row"])
    sut = cge.SupplyUse(
        make=block(sam, acts, coms), use=block(sam, coms, acts), final=final,
        imports=block(sam, ["row"], coms).iloc[0],
        value_added=block(sam, facs, acts),
        product_taxes=block(sam, ["gov"], coms).iloc[0],
        production_taxes=block(sam, ["gov"], acts).iloc[0],
        margins=block(sam, coms, coms) if margins else None,
        tariffs=block(sam, ["trf"], coms).iloc[0] if trade_taxes else None,
        output_taxes=block(sam, ["idt"], coms).iloc[0] if trade_taxes else None,
        direct_sales=block(sam, acts, hhs) if own else None)
    inst = cge.Institutions(factor_income=block(sam, hhs + ["gov", "row"], facs))
    if institutions:
        inst = cge.Institutions(
            factor_income=block(sam, hhs + ["gov", "row"], facs),
            direct_taxes=block(sam, ["gov"], hhs).iloc[0],
            transfers=block(sam, hhs, ["gov"]).iloc[:, 0],
            from_abroad=block(sam, hhs, ["row"]).iloc[:, 0],
            to_abroad=block(sam, ["row"], hhs).iloc[0])
    return sut, inst


CASES = {
    "plain": (toy.build, {}),
    "margins": (toy.build_with_margins, dict(margins=True)),
    "trade_taxes": (toy.build_with_trade_taxes, dict(trade_taxes=True)),
    "own_account": (toy.build_with_own_production, dict(own=True)),
    "institutions": (toy.build_with_institutions, dict(institutions=True)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_tables_give_back_the_sam(case):
    build, kw = CASES[case]
    original, _ = build()
    sut, inst = tables(original, **kw)
    rebuilt = cge.sut_to_sam(sut, inst)
    assert sorted(rebuilt.accounts) == sorted(original.accounts)
    accounts = original.accounts
    np.testing.assert_allclose(rebuilt.matrix(accounts).to_numpy(),
                               original.matrix(accounts).to_numpy(), atol=1e-12)
    rebuilt.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    g = sut.gdp()
    assert g["production"] == pytest.approx(g["expenditure"], rel=1e-12)
    assert g["income"] == pytest.approx(g["production"], rel=1e-12)


def test_the_rebuilt_sam_calibrates_to_the_same_base_year():
    original, econ = toy.build_with_institutions()
    sut, inst = tables(original, institutions=True)
    rebuilt = cge.sut_to_sam(sut, inst)
    econ2 = cge.calibrate(rebuilt, toy.spec())
    rep = cge.base_year_reproduction(econ2)
    assert rep["ok"], rep["checks"]
    np.testing.assert_allclose(econ2.hh_income0, econ.hh_income0, rtol=1e-14)


def test_the_provenance_of_every_block():
    original, _ = toy.build_with_institutions()
    sut, inst = tables(original, institutions=True)
    sam = cge.sut_to_sam(sut, inst, source="toy tables")
    p = sam.provenance.set_index("note")
    assert p.at["output, basic prices", "source"] == "toy tables"
    assert p.at["direct taxes", "source"] == "integrated economic accounts"
    assert p.at["household saving", "source"] == "residual"
    assert p.at["foreign saving", "total"] == pytest.approx(18.0)       # the toy's docstring
    assert p.at["household saving", "total"] == pytest.approx(7.0 + 10.0)
    assert p.at["government saving", "total"] == pytest.approx(15.0)
    # Every cell of the SAM is in one block.
    assert int(sam.provenance["cells"].sum()) == int((sam.frame["value"] != 0).sum())


def test_one_household_takes_all_factor_income_by_default():
    sam, _ = toy.build()
    sut, _ = tables(sam)
    one = sut.final.drop(columns=["hh_2"]).rename(columns={"hh_1": "hh"})
    one["hh"] += sut.final["hh_2"]
    sut1 = cge.SupplyUse(make=sut.make, use=sut.use, final=one, imports=sut.imports,
                         value_added=sut.value_added, product_taxes=sut.product_taxes,
                         production_taxes=sut.production_taxes)
    built = cge.sut_to_sam(sut1)
    assert built.matrix().at["hh", "f_lab"] == pytest.approx(110.0)
    p = built.provenance.set_index("note")
    assert p.at["factor income", "source"] == "assumed: all factor income to households"
    with pytest.raises(ValueError, match="several household accounts"):
        cge.sut_to_sam(sut)


def test_tables_that_do_not_add_up_are_refused():
    sam, _ = toy.build_with_margins()
    sut, inst = tables(sam, margins=True)
    use = sut.use.copy()
    use.loc["c_agr", "a_ind"] += 1.0
    with pytest.raises(ValueError, match="c_agr"):
        cge.sut_to_sam(cge.SupplyUse(make=sut.make, use=use, final=sut.final,
                                     imports=sut.imports, value_added=sut.value_added,
                                     product_taxes=sut.product_taxes,
                                     production_taxes=sut.production_taxes,
                                     margins=sut.margins), inst)
    va = sut.value_added.copy()
    va.loc["f_lab", "a_trd"] += 1.0
    with pytest.raises(ValueError, match="a_trd"):
        cge.sut_to_sam(cge.SupplyUse(make=sut.make, use=sut.use, final=sut.final,
                                     imports=sut.imports, value_added=va,
                                     product_taxes=sut.product_taxes,
                                     production_taxes=sut.production_taxes,
                                     margins=sut.margins), inst)
    margins = sut.margins.copy()
    margins.loc["c_trd", "c_trd"] = -11.0
    margins.loc["c_trd", "c_agr"] = 4.0                     # still balanced by commodity? no
    with pytest.raises(ValueError):
        cge.sut_to_sam(cge.SupplyUse(make=sut.make, use=sut.use, final=sut.final,
                                     imports=sut.imports, value_added=sut.value_added,
                                     product_taxes=sut.product_taxes,
                                     production_taxes=sut.production_taxes,
                                     margins=margins), inst)
    fi = inst.factor_income.copy()
    fi.loc["hh_1", "f_cap"] += 1.0
    with pytest.raises(ValueError, match="differs from value added"):
        cge.sut_to_sam(sut, cge.Institutions(factor_income=fi))


def test_flows_between_institutions_and_flows_that_unbalance_an_account():
    sam, _ = toy.build_with_institutions()
    sut, inst = tables(sam, institutions=True)
    # A grant from abroad to government moves saving between them and changes
    # no total: government saves 2 more, foreign saving is 2 less.
    inst.flows = [("gov", "row", 2.0, "a grant")]
    built = cge.sut_to_sam(sut, inst)
    built.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    p = built.provenance.set_index("note")
    assert p.at["other flow", "source"] == "a grant"
    assert p.at["government saving", "total"] == pytest.approx(17.0)
    assert p.at["foreign saving", "total"] == pytest.approx(16.0)
    # A payment to a commodity with no counterpart leaves the commodity unbalanced.
    inst.flows = [("c_ind", "gov", 2.0, "an unrecorded purchase")]
    with pytest.raises(ValueError, match="c_ind"):
        cge.sut_to_sam(sut, inst)


def test_misuse_is_refused():
    sam, _ = toy.build()
    sut, inst = tables(sam)
    with pytest.raises(KeyError, match="not commodities"):
        cge.SupplyUse(make=sut.make, use=sut.use, final=sut.final,
                      imports=pd.Series({"c_oil": 1.0}), value_added=sut.value_added)
    with pytest.raises(ValueError, match="role"):
        cge.SupplyUse(make=sut.make, use=sut.use, final=sut.final, imports=sut.imports,
                      value_added=sut.value_added, final_roles={"hh_1": "consumers"})
    fi = inst.factor_income.rename(index={"hh_1": "hh_9"})
    with pytest.raises(KeyError, match="neither"):
        cge.sut_to_sam(sut, cge.Institutions(factor_income=fi))
    with pytest.raises(TypeError, match="DataFrame"):
        cge.SupplyUse(make=sut.make.to_numpy(), use=sut.use, final=sut.final,
                      imports=sut.imports, value_added=sut.value_added)
