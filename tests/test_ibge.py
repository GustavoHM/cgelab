"""IBGE readers: the SAS reading programs, the PNAD Contínua and the POF.

The fixtures in ``tests/data/ibge`` are written by hand in the IBGE's formats
(Latin-1, CRLF, accented comments, several data steps in one program, ``w.d``
informats, a ``;`` glued to the last field), with positions of their own: the
microdata do not enter the repository. Every expected number below was worked
out from those files on paper. A check on the real microdata runs only when
``CGELAB_IBGE_DIR`` points at them.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cgelab as cge
from cgelab import ibge
from cgelab.examples import toy

DATA = Path(__file__).resolve().parent / "data" / "ibge"
PNADC = (DATA / "PNADC_teste.txt", DATA / "pnadc_input.txt")
POF_PROGRAM = DATA / "pof_leitura.txt"
SEGMENTS = {"formal": "f_for", "informal": "f_inf"}
COMMODITIES = {"00001": "c_ser", "100": "c_ser", "07": "c_ind", "63": "c_agr", "65": "c_agr",
               "6500": "c_ind", "44": "c_ind"}


def two_accounts(households):
    return np.where(households["household"].str.startswith("11"), "hh_1", "hh_2")


# -- the SAS program ----------------------------------------------------------

def test_the_sas_program_of_one_record():
    records = ibge.read_sas_program(PNADC[1])
    assert list(records) == ["pnadc_teste"]
    r = records["pnadc_teste"]
    assert r.infile == "PNADC_teste.txt"
    f = r.fields.set_index("name")
    assert f.loc["Ano"].tolist() == [1, 4, True, 0]
    assert f.loc["V1028"].tolist() == [19, 15, False, 0]
    assert f.loc["V9999"].tolist() == [55, 4, False, 2]
    assert f.loc["V4013"].tolist() == [59, 5, True, 0]
    assert len(f) == 17


def test_the_sas_program_of_several_records():
    records = ibge.read_sas_program(POF_PROGRAM)
    assert sorted(records) == sorted(["morador", "aluguel_estimado", "despesa_coletiva",
                                      "caderneta_coletiva", "despesa_individual",
                                      "rendimento_trabalho", "outros_rendimentos"])
    m = records["morador"].fields.set_index("name")
    assert m.loc["PESO_FINAL"].tolist() == [15, 14, False, 8]          # the last field, ';'
    d = records["despesa_individual"]
    assert d.infile == "DESPESA_INDIVIDUAL.txt"
    assert d.fields.set_index("name").loc["V8000_DEFLA", "decimals"] == 2


def test_reading_fixed_width_data():
    f = ibge.read_ibge(*PNADC)
    assert len(f) == 8
    assert f["Ano"].tolist() == ["2023"] * 8                            # text stays text
    assert f["UPA"].iloc[0] == "110000016"
    assert f["V1028"].iloc[0] == pytest.approx(1234.567890123, rel=1e-12)
    assert f["V2009"].tolist() == [45, 42, 19, 10, 30, 70, 35, 25]
    # A w.d informat: the point d digits from the right when the field has none.
    assert f["V9999"].iloc[0] == pytest.approx(12.34, rel=1e-14)
    assert f["V9999"].iloc[1] == pytest.approx(12.5, rel=1e-14)
    assert np.isnan(f["V9999"].iloc[2])                                  # blank is missing
    assert pd.isna(f["V4019"].iloc[0])                                  # blank text too
    sub = ibge.read_ibge(*PNADC, variables=["UPA", "VD4019"])
    assert list(sub.columns) == ["UPA", "VD4019"]
    chunks = ibge.read_ibge(*PNADC, chunksize=3)
    pd.testing.assert_frame_equal(chunks, f)


def test_choosing_a_record():
    r = ibge.read_ibge(DATA / "DESPESA_COLETIVA.txt", POF_PROGRAM)       # found by its INFILE
    assert r["V8000_DEFLA"].tolist() == [100.0, 240.0]
    with pytest.raises(ValueError, match="say which"):
        ibge.read_ibge(DATA / "PNADC_teste.txt", POF_PROGRAM)
    with pytest.raises(KeyError, match="not in the program"):
        ibge.read_ibge(DATA / "MORADOR.txt", POF_PROGRAM, record="domicilio")
    with pytest.raises(KeyError, match="not in the layout"):
        ibge.read_ibge(*PNADC, variables=["V9998"])


# -- the PNAD Contínua --------------------------------------------------------

def test_the_pnad_continua():
    survey, persons = ibge.pnadc(*PNADC, segments=SEGMENTS, group=two_accounts)
    p = persons.frame
    assert p["status"].tolist() == ["f_for", "f_inf", "unemployed", "inactive",
                                    "f_inf", "inactive", "f_for", "f_inf"]
    # Own-account without CNPJ (09, V4019 = 2) and the unpaid family worker (10)
    # are informal; the employer with CNPJ (08, V4019 = 1) is formal.
    assert p["earnings"].tolist() == [3000.0, 1200.0, 0, 0, 900.0, 0, 5000.0, 0]
    assert p["female"].tolist() == [0, 1, 0, 1, 1, 0, 0, 1]
    assert p["education"].tolist() == [11, 9, 12, 4, 8, 3, 15, 10]
    h = survey.frame.set_index("household")
    assert h.loc["1100000160101", "weight"] == pytest.approx(1234.567890123, rel=1e-12)
    assert h.loc["1100000160101", ["formal", "informal", "size"]].tolist() == [3000, 1200, 4]
    assert h.loc["2900000270305", ["formal", "informal", "size"]].tolist() == [5000, 900, 4]
    assert h["group"].tolist() == ["hh_1", "hh_2"]
    assert survey.sources == {"formal": "f_for", "informal": "f_inf"}
    # The activity of the main job, for the occupied only.
    assert p["activity_code"].tolist() == ["25001", "01101", "", "", "97000", "", "47001",
                                           "01102"]


def test_the_activity_of_each_worker():
    acts = {"01": "a_agr", "2": "a_ind", "4": "a_ind", "97": "a_ind"}
    _, persons = ibge.pnadc(*PNADC, segments=SEGMENTS, group=two_accounts, activities=acts)
    assert persons.frame["activity"].fillna("").tolist() == ["a_ind", "a_agr", "", "", "a_ind",
                                                             "", "a_ind", "a_agr"]
    with pytest.raises(ValueError, match="97000"):
        ibge.pnadc(*PNADC, segments=SEGMENTS, group=two_accounts,
                   activities={"01": "a_agr", "2": "a_ind", "4": "a_ind"})


def test_the_cnpj_before_2015():
    f = ibge.read_ibge(*PNADC)
    f.loc[1, "V4019"] = None                                 # an own-account worker, no V4019
    with pytest.raises(ValueError, match="missing_cnpj"):
        ibge.pnadc(None, None, frame=f, segments=SEGMENTS, group=two_accounts)
    _, informal = ibge.pnadc(None, None, frame=f, segments=SEGMENTS, group=two_accounts,
                             missing_cnpj="informal")
    _, formal = ibge.pnadc(None, None, frame=f, segments=SEGMENTS, group=two_accounts,
                           missing_cnpj="formal")
    assert informal.frame["status"].iloc[1] == "f_inf"
    assert formal.frame["status"].iloc[1] == "f_for"


def test_the_pnad_continua_feeds_the_microsimulation():
    survey, persons = ibge.pnadc(*PNADC, segments=SEGMENTS, group=two_accounts)
    _, econ = toy.build_with_segmented_labour()
    floor = cge.LabourMarket(rigid={"f_for": 1.1}, informal="f_inf")
    ref = cge.solve_period(econ)
    alt = cge.solve_period(econ, cge.Closure(labour_markets={"labour": floor}))
    res = cge.microsimulate(survey, ref, alt, reconcile=False, reweight=False)
    A_ref, A_alt = cge.account_sources(ref), cge.account_sources(alt)
    for h, seg in (("hh_1", "f_for"), ("hh_2", "f_inf")):
        m = (res.frame["group"] == h).to_numpy()
        assert res.income_alt.loc[m, seg].sum() == pytest.approx(
            res.income_ref.loc[m, seg].sum() * A_alt.at[h, seg] / A_ref.at[h, seg], rel=1e-12)


def test_pnad_continua_misuse():
    with pytest.raises(ValueError, match="'formal' and 'informal'"):
        ibge.pnadc(*PNADC, segments={"formal": "f_for"}, group=two_accounts)
    with pytest.raises(ValueError, match="missing_cnpj must"):
        ibge.pnadc(*PNADC, segments=SEGMENTS, group=two_accounts, missing_cnpj="maybe")


# -- the POF ------------------------------------------------------------------

def test_the_pof():
    survey = ibge.pof(DATA, POF_PROGRAM, commodities=COMMODITIES, income={"540": "transfers"},
                      group=lambda h: ["hh_1"] * len(h))
    f = survey.frame.set_index("household")
    uc1, uc2, uc3 = "110000001-1-1", "110000001-1-2", "290000002-3-1"
    # Residents of each consumption unit, the domestic employee (V0306 = 18) left out.
    assert f["size"].tolist() == [2.0, 1.0, 3.0]
    assert f.loc[uc1, "weight"] == 250.5
    # Monthly spending: V8000_DEFLA x FATOR_ANUALIZACAO / 12, times V9011 where monthly.
    assert f.loc[uc1, "spend_c_ser"] == pytest.approx(500 * 12 / 12 + 100 * 3 * 4 / 12)
    assert f.loc[uc1, "unmapped_spending"] == pytest.approx(120 / 12)  # quadro 31, not monthly
    assert f.loc[uc2, "spend_c_agr"] == pytest.approx(30 * 52 / 12)
    # 07... at 240/12, 6500... (the longest prefix) at 12*52/12, quadro 44 with its months.
    assert f.loc[uc3, "spend_c_ind"] == pytest.approx(240 / 12 + 12 * 52 / 12 + 50 * 6 * 2 / 12)
    assert f.loc[uc3, "spend_c_agr"] == 0.0
    # Work income always with V9011; other income only for quadro 54.
    assert f.loc[uc1, "labour"] == pytest.approx(3000 * 12 / 12)
    assert f.loc[uc3, "labour"] == pytest.approx(1000 * 6 / 12)          # the blank row is out
    assert f.loc[uc2, "transfers"] == pytest.approx(900 * 12 / 12)
    assert f.loc[uc3, "other_income"] == pytest.approx(1200 / 12)
    assert survey.budget == {"spend_c_agr": "c_agr", "spend_c_ind": "c_ind",
                             "spend_c_ser": "c_ser"}
    assert set(survey.sources.values()) == {"fixed"}


def test_the_pof_counts_whom_it_is_told_to():
    survey = ibge.pof(DATA, POF_PROGRAM, commodities=COMMODITIES, exclude_conditions=(),
                      group=lambda h: ["hh_1"] * len(h))
    assert survey.frame["size"].tolist() == [3.0, 1.0, 3.0]


# -- the real microdata, when they are there ----------------------------------

IBGE_DIR = os.environ.get("CGELAB_IBGE_DIR")


@pytest.mark.skipif(IBGE_DIR is None, reason="set CGELAB_IBGE_DIR to the IBGE microdata")
def test_a_real_quarter_of_the_pnad_continua():
    root = Path(IBGE_DIR)
    data = sorted(root.glob("PNADC_0*.txt"))
    program = root / "input_PNADC_trimestral.txt"
    if not data or not program.exists():
        pytest.skip("no PNADC_0*.txt and input_PNADC_trimestral.txt in CGELAB_IBGE_DIR")
    survey, persons = ibge.pnadc(data[0], program, segments=SEGMENTS,
                                 group=lambda h: np.full(len(h), "hh_1"))
    f, p = survey.frame, persons.frame
    population = float((f["weight"] * f["size"]).sum())
    assert 1.8e8 < population < 2.4e8                        # Brazil's resident population
    working = p["status"].isin(["f_for", "f_inf"])
    w = f.loc[p["household"], "weight"].to_numpy()
    informal = (w * (p["status"] == "f_inf")).sum() / (w * working).sum()
    assert 0.25 < informal < 0.55                            # the IBGE's rate has been ~40%
