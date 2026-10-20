"""The IBGE's supply and use tables, read into a SAM.

The fixture is written by hand in the layout of the IBGE's tables 1 and 2 (the
sheets ``oferta``, ``producao``, ``importacao``, ``CI``, ``demanda``, ``VA``,
the title rows, the codes in the first column, the activity codes at the top of
each header cell, the total row) for three activities and three products: a
good, trade and transport. Its numbers were closed on paper: every product's
supply at purchaser prices equals its total demand, every activity's output its
costs, and GDP is 117 by production, expenditure and income. A check on the
IBGE's published 2021 table, against the GDP the IBGE publishes for that year,
runs when ``CGELAB_IBGE_DIR`` holds it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cgelab as cge

PRODUCTS = [("01911", "Arroz, trigo e outros cereais"), ("46801", "Comércio por atacado"),
            ("49001", "Transporte terrestre")]
ACTIVITIES = ["0191\nAgricultura", "4680\nComércio", "4900\nTransporte terrestre"]


def sheet(title, top, heads, rows, total=None, notes=()):
    """An IBGE sheet: title, blank, two header rows, blank, product rows, blank, total."""
    width = 2 + len(heads)
    lines = [[title] + [""] * (width - 1), [""] * width,
             ["Código\ndo\nproduto", "Descrição do produto", top] + [""] * (width - 3),
             ["", ""] + heads, [""] * width]
    lines += [[code, name] + [str(v) for v in vals] for (code, name), vals in rows]
    lines += [[""] * width]
    if total is not None:
        lines += [["Total", ""] + [str(v) for v in total]]
    lines += [[n] + [""] * (width - 1) for n in notes]
    return pd.DataFrame(lines)


def va_sheet(rows):
    width = 1 + len(ACTIVITIES) + 1
    lines = [["Tabela 2 - Usos de bens e serviços"] + [""] * (width - 1), [""] * width,
             ["Operações", "Componentes do valor adicionado"] + [""] * (width - 2),
             [""] + ACTIVITIES + ["Total do produto"], [""] * width]
    lines += [[label] + [str(v) for v in vals] + [str(sum(vals))] for label, vals in rows]
    return pd.DataFrame(lines)


OFERTA = ["Oferta total\na preço\nde consumidor", "Margem \nde\ncomércio", "Margem\nde\ntransporte",
          "Imposto\nde\nimportação", "IPI", "ICMS", "Outros impostos menos subsídios",
          "Total de impostos líquidos de subsídios", "Oferta total\n a preço básico"]


def fixture():
    supply = {
        "oferta": sheet("Tabela 1 - Recursos de bens e serviços", "Oferta de bens e serviços",
                        OFERTA, list(zip(PRODUCTS, [[133, 12, 5, 1, 2, 3, 0, 6, 110],
                                                    [19, -12, 0, 0, 0, 1, 0, 1, 30],
                                                    [16, 0, -5, 0, 0, 0, 1, 1, 20]])),
                        [168, 0, 0, 1, 2, 4, 1, 8, 160]),
        "producao": sheet("Tabela 1 - Recursos de bens e serviços", "Produção das atividades",
                          ACTIVITIES + ["Total do produto"],
                          list(zip(PRODUCTS, [[100, 0, 0, 100], [0, 30, 0, 30], [0, 0, 20, 20]])),
                          [100, 30, 20, 150]),
        "importacao": sheet("Tabela 1 - Recursos de bens e serviços", "Valores correntes",
                            ["Importação de bens\ne serviços (1)"],
                            list(zip(PRODUCTS, [[10], [0], [0]])), [10],
                            notes=["Fonte: IBGE", "(1) líquida de ajuste CIF/FOB."]),
    }
    use = {
        "CI": sheet("Tabela 2 - Usos de bens e serviços", "Consumo intermediário",
                    ACTIVITIES + ["Total do produto"],
                    list(zip(PRODUCTS, [[20, 5, 3, 28], [4, 2, 1, 7], [3, 1, 2, 6]])),
                    [27, 8, 6, 41]),
        "demanda": sheet("Tabela 2 - Usos de bens e serviços", "Demanda final",
                         ["Exportação de bens e serviços (1)", "Consumo do governo",
                          "Consumo das ISFLSF", "Consumo das famílias",
                          "Formação bruta de capital fixo", "Variação de estoque",
                          "Demanda final", "Demanda total"],
                         list(zip(PRODUCTS, [[15, 0, 0, 80, 6, 4, 105, 133],
                                             [0, 0, 2, 10, 0, 0, 12, 19],
                                             [2, 3, 0, 5, 0, 0, 10, 16]]))),
        "VA": va_sheet([("Valor adicionado bruto ( PIB )", [73, 22, 14]),
                        ("Remunerações", [30, 12, 8]), ("Salários", [25, 10, 7]),
                        ("Excedente operacional bruto e rendimento misto bruto", [40, 10, 5]),
                        ("Rendimento misto bruto", [15, 2, 1]),
                        ("Excedente operacional bruto (EOB)", [25, 8, 4]),
                        ("Outros impostos sobre a produção", [4, 0, 1]),
                        ("Outros subsídios à produção", [-1, 0, 0]),
                        ("Valor da produção", [100, 30, 20]),
                        ("Fator trabalho (ocupações)", [500, 90, 40])]),
    }
    return supply, use


def test_reading_the_tables():
    sut = cge.read_tru(*fixture())
    assert sut.activities == ["a_0191", "a_4680", "a_4900"]
    assert sut.commodities == ["c_01911", "c_46801", "c_49001"]
    assert sut.labels["a_4680"] == "Comércio"
    assert sut.labels["c_49001"] == "Transporte terrestre"
    assert sut.make.at["a_0191", "c_01911"] == 100.0
    assert sut.use.at["c_46801", "a_0191"] == 4.0
    assert sut.imports.tolist() == [10.0, 0.0, 0.0]
    # Labour, capital and mixed income, apart; the occupations row is not money.
    va = sut.value_added
    assert va.loc["f_lab"].tolist() == [30.0, 12.0, 8.0]
    assert va.loc["f_cap"].tolist() == [25.0, 8.0, 4.0]
    assert va.loc["f_mix"].tolist() == [15.0, 2.0, 1.0]
    assert sut.production_taxes.tolist() == [3.0, 0.0, 1.0]           # net of subsidies
    assert sut.product_taxes.tolist() == [6.0, 1.0, 1.0]              # II + IPI + ICMS + others
    # Final demand: NPISH with households, FBCF and stocks into investment.
    assert sut.final.loc["c_46801", "hh"] == 12.0
    assert sut.final["s_i"].tolist() == [10.0, 0.0, 0.0]
    assert sut.final["row"].tolist() == [15.0, 0.0, 2.0]
    assert sut.final["gov"].tolist() == [0.0, 0.0, 3.0]
    # Margins: each good's margin goes to the products that provide it.
    m = sut.margins
    assert m.loc["c_46801"].tolist() == [12.0, -12.0, 0.0]
    assert m.loc["c_49001"].tolist() == [5.0, 0.0, -5.0]


def test_the_tables_build_a_sam_whose_three_gdps_agree():
    sut = cge.read_tru(*fixture())
    assert sut.gdp() == {"production": 117.0, "expenditure": 117.0, "income": 117.0}
    inst = cge.Institutions(factor_income=pd.DataFrame(
        {"f_lab": [50.0], "f_cap": [37.0], "f_mix": [18.0]}, index=["hh"]))
    sam = cge.sut_to_sam(sut, inst, source="IBGE TRU (fixture)")
    sam.check_balance(rel_tol=1e-12, abs_tol=1e-9)
    p = sam.provenance.set_index("note")
    assert p.at["margins shared among their providers in proportion to what each provides",
                "source"] == "assumed by read_tru"
    assert "mixed income: separate" in p.index
    econ = cge.calibrate(sam, cge.ModelSpec(factors=("f_lab", "f_cap", "f_mix"), households=["hh"]))
    rep = cge.base_year_reproduction(econ)
    assert rep["ok"], rep["checks"]


def test_the_choices_the_reader_makes_explicit():
    supply, use = fixture()
    sut = cge.read_tru(supply, use, mixed_income="labour", npish="government", tariffs=True)
    assert sut.value_added.loc["f_lab"].tolist() == [45.0, 14.0, 9.0]
    assert "f_mix" not in sut.value_added.index
    assert sut.final.loc["c_46801", ["hh", "gov"]].tolist() == [10.0, 2.0]
    assert sut.tariffs.tolist() == [1.0, 0.0, 0.0]
    assert sut.product_taxes.tolist() == [5.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="mixed_income"):
        cge.read_tru(supply, use, mixed_income="profits")
    with pytest.raises(ValueError, match="npish"):
        cge.read_tru(supply, use, npish="firms")


def test_a_table_that_does_not_close_is_refused():
    supply, use = fixture()
    use["CI"].iat[5, 2] = "21"                          # one more unit of cereal used by farming
    sut = cge.read_tru(supply, use)
    with pytest.raises(ValueError, match="c_01911"):
        cge.sut_to_sam(sut)


IBGE_DIR = os.environ.get("CGELAB_IBGE_DIR")


@pytest.mark.skipif(IBGE_DIR is None, reason="set CGELAB_IBGE_DIR to the IBGE's tables")
def test_the_published_2021_table():
    pytest.importorskip("xlrd")
    root = Path(IBGE_DIR)
    t1, t2 = root / "68_tab1_2021.xls", root / "68_tab2_2021.xls"
    if not (t1.exists() and t2.exists()):
        pytest.skip("68_tab1_2021.xls and 68_tab2_2021.xls are not in CGELAB_IBGE_DIR")
    sut = cge.read_tru(t1, t2)
    assert len(sut.activities) == 68 and len(sut.commodities) == 128
    # The IBGE publishes 2021 GDP as R$ 9,012.142 billion (synoptic table 1).
    g = sut.gdp()
    assert g["production"] == pytest.approx(9_012_142.0, abs=0.5)
    assert g["expenditure"] == pytest.approx(9_012_142.0, abs=0.5)
    sam = cge.sut_to_sam(sut)
    sam.check_balance(rel_tol=1e-9, abs_tol=1e-3)
