"""Readers for the IBGE's household surveys: the PNAD Contínua and the POF 2017–2018.

The IBGE publishes its microdata as fixed-width text, with a SAS reading
program that gives every field's position, width and type
(``@0001 Ano $4. /* Ano de referência */``). :func:`read_ibge` reads that
program and the data it describes, so no position is written by hand here and
a new release of the layout needs no new code.

:func:`pnadc` turns a quarter of the PNAD Contínua into a household
:class:`cgelab.micro.Survey` and a :class:`cgelab.behaviour.PersonSurvey`, with
each worker formal or informal by the IBGE's informality proxy. :func:`pof`
turns the POF 2017–2018 into a household survey of consumption units, with their
spending, annualised and deflated as the IBGE's own tabulation programs do, in
the commodities of a SAM through a correspondence the user supplies.

The definitions below were checked against the IBGE's documentation: the
dictionary and SAS program of the PNAD Contínua (Dicionario_e_input_20221031),
the definition of its derived variables (part 4, labour), and the POF's
dictionary, reading programs and calculation memo (Memoria_de_Calculo_20230929).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["SasLayout", "read_sas_program", "read_ibge", "pnadc", "pof", "INFORMAL_VD4009",
           "POF_MONTHLY_QUADROS"]

_FIELD = re.compile(r"@\s*(\d+)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\$?)(\d+)\.(\d*)")
_DATA = re.compile(r"^\s*data\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.IGNORECASE)
_INFILE = re.compile(r'infile\s+"([^"]*)"', re.IGNORECASE)


@dataclass
class SasLayout:
    """One record of a SAS reading program: its data file and its fields.

    ``fields`` has one row per variable: ``name``, ``start`` (1-based),
    ``width``, ``char`` (a ``$`` informat) and ``decimals`` (the ``d`` of a
    ``w.d`` informat).
    """

    name: str
    infile: str
    fields: pd.DataFrame


def read_sas_program(path, encoding: str = "latin-1") -> dict:
    """The records of an IBGE SAS reading program, by data-step name.

    The PNAD Contínua has one record; the POF one per file
    (``domicilio``, ``morador``, ``despesa_individual``...). Each record's
    ``infile`` is the base name of the data file its ``INFILE`` statement names.
    """
    with open(path, encoding=encoding, newline="") as fh:
        text = fh.read()
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)        # comments, which may hold '@'
    records, name, infile, rows = {}, None, "", []

    def close():
        if name is not None and rows:
            records[name] = SasLayout(name=name, infile=infile,
                                      fields=pd.DataFrame(rows, columns=["name", "start", "width",
                                                                         "char", "decimals"]))

    for line in text.splitlines():
        m = _DATA.match(line)
        if m:
            close()
            name, infile, rows = m.group(1).lower(), "", []
            continue
        m = _INFILE.search(line)
        if m:
            infile = re.split(r"[\\/]", m.group(1))[-1]
            continue
        for m in _FIELD.finditer(line):
            rows.append((m.group(2), int(m.group(1)), int(m.group(4)), m.group(3) == "$",
                         int(m.group(5) or 0)))
    close()
    if not records:
        raise ValueError(f"no INPUT fields found in {path}: is it an IBGE SAS reading program?")
    return records


def read_ibge(data, program, *, record: str | None = None,
              variables: Sequence[str] | None = None, encoding: str = "latin-1",
              chunksize: int | None = None) -> pd.DataFrame:
    """Read IBGE fixed-width microdata with the layout of its SAS reading program.

    ``record`` names the data step when the program has several (the POF's);
    without it, the one whose ``INFILE`` has the data file's base name, or the
    only one. ``variables`` keeps a subset, which is what makes a quarter of the
    PNAD Contínua fit in memory. Character fields stay strings; numeric fields
    become numbers, blanks become NaN, and a ``w.d`` informat puts the decimal
    point ``d`` digits from the right when the field has none, as SAS does.
    """
    records = read_sas_program(program, encoding=encoding)
    if record is None:
        base = os.path.basename(str(data)).lower()
        match = [r for r in records.values() if r.infile.lower() == base]
        if len(match) == 1:
            layout = match[0]
        elif len(records) == 1:
            layout = next(iter(records.values()))
        else:
            raise ValueError(f"the program has records {sorted(records)}; say which with record=")
    else:
        if record.lower() not in records:
            raise KeyError(f"record {record!r} is not in the program ({sorted(records)})")
        layout = records[record.lower()]
    fields = layout.fields
    if variables is not None:
        missing = [v for v in variables if v not in set(fields["name"])]
        if missing:
            raise KeyError(f"variables not in the layout of {layout.name!r}: {missing}")
        fields = fields[fields["name"].isin(variables)]
    colspecs = [(s - 1, s - 1 + w) for s, w in zip(fields["start"], fields["width"])]
    reader = pd.read_fwf(data, colspecs=colspecs, names=list(fields["name"]), dtype=str,
                         encoding=encoding, header=None, chunksize=chunksize,
                         keep_default_na=False)
    frames = [reader] if chunksize is None else reader
    out = []
    for frame in frames:
        for _, f in fields.iterrows():
            col = frame[f["name"]].str.strip()
            if f["char"]:
                frame[f["name"]] = col.where(col != "", None)
                continue
            num = pd.to_numeric(col.where(col != "", None), errors="coerce")
            if f["decimals"]:
                implied = col.notna() & (col != "") & ~col.str.contains(".", regex=False)
                num = num.where(~implied, num / 10 ** f["decimals"])
            frame[f["name"]] = num
        out.append(frame)
    return out[0] if len(out) == 1 else pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# PNAD Contínua
# ---------------------------------------------------------------------------

#: VD4009 categories the IBGE counts as informal: private-sector employee and
#: domestic worker without a signed card (02, 04) and unpaid family worker
#: (10). Employers (08) and own-account workers (09) are informal when their
#: business has no CNPJ (V4019 = 2).
INFORMAL_VD4009 = ("02", "04", "10")
_CNPJ_VD4009 = ("08", "09")

_PNADC_VARS = ["Ano", "Trimestre", "UPA", "V1008", "V1014", "V1028", "V2005", "V2007",
               "V2009", "V4013", "V4019", "VD3005", "VD4001", "VD4002", "VD4009", "VD4019"]


def pnadc(data, program, *, segments: Mapping[str, str], group, frame: pd.DataFrame | None = None,
          activities: Mapping[str, str] | None = None, missing_cnpj: str = "refuse",
          chunksize: int | None = 200_000) -> tuple:
    """A quarter of the PNAD Contínua as a household ``Survey`` and a ``PersonSurvey``.

    ``segments`` maps ``"formal"`` and ``"informal"`` to the model's labour
    segments. A person of 14 or more out of the labour force (VD4001 = 2), or
    under 14, is ``"inactive"``; unoccupied (VD4002 = 2), ``"unemployed"``;
    occupied, informal by the IBGE's proxy (VD4009 02, 04 or 10, or 08 and 09
    without CNPJ) and formal otherwise. Earnings are VD4019, usual monthly
    earnings from all jobs. Before the fourth quarter of 2015 there is no
    V4019, so ``missing_cnpj`` says whether employers and own-account workers
    without it are ``"formal"`` or ``"informal"``; by default they are refused.

    ``group`` assigns each household to a SAM account: a column of the
    household frame, or a function of it (it has ``household``, ``weight``,
    ``size``, ``formal``, ``informal``). Characteristics of the persons are
    ``education`` (VD3005, years), ``female`` and ``age``. ``frame`` passes the
    person rows already read, instead of ``data`` and ``program``.

    Each occupied person's activity is V4013, the CNAE-Domiciliar code of the
    main job, kept in ``activity_code``. ``activities`` maps a prefix of that
    code to a SAM activity (the longest prefix wins) into ``activity``; an
    occupied person whose code no prefix maps is refused, with the codes, since
    their earnings would otherwise be split among activities by nobody's rule.
    """
    from .behaviour import PersonSurvey
    from .micro import Survey

    if set(segments) != {"formal", "informal"}:
        raise ValueError("segments must map 'formal' and 'informal' to the model's segments")
    if missing_cnpj not in ("refuse", "formal", "informal"):
        raise ValueError("missing_cnpj must be 'refuse', 'formal' or 'informal'")
    p = frame.copy() if frame is not None else read_ibge(data, program, variables=_PNADC_VARS,
                                                          chunksize=chunksize)
    missing = [v for v in _PNADC_VARS if v not in p.columns]
    if missing:
        raise KeyError(f"PNAD Contínua variables missing: {missing}")
    def code(column, width=1):
        """A coded variable as text, a blank (not applicable) as the empty string."""
        text = p[column].fillna("").astype(str).str.strip()
        text = text.str.replace(r"\.0$", "", regex=True)          # read as a number somewhere
        return text.where(text == "", text.str.zfill(width)).to_numpy()

    p["household"] = code("UPA") + code("V1008", 2) + code("V1014", 2)
    age = p["V2009"].astype(float)
    vd4009 = pd.Series(code("VD4009", 2))
    occupied = code("VD4002") == "1"
    no_cnpj = code("V4019") == "2"
    cnpj_unknown = code("V4019") == ""
    informal = occupied & (vd4009.isin(INFORMAL_VD4009).to_numpy()
                           | (vd4009.isin(_CNPJ_VD4009).to_numpy() & no_cnpj))
    unknown = occupied & vd4009.isin(_CNPJ_VD4009).to_numpy() & cnpj_unknown
    if unknown.any():
        if missing_cnpj == "refuse":
            raise ValueError(
                f"{int(unknown.sum())} employers or own-account workers have no V4019 (CNPJ), "
                f"which the survey asks only from the fourth quarter of 2015; say how to "
                f"classify them with missing_cnpj='formal' or 'informal'")
        if missing_cnpj == "informal":
            informal = informal | unknown
    status = np.where(age < 14, "inactive",
                      np.where(code("VD4001") == "2", "inactive",
                               np.where(code("VD4002") == "2", "unemployed",
                                        np.where(occupied,
                                                 np.where(informal, segments["informal"],
                                                          segments["formal"]),
                                                 "inactive"))))
    earnings = p["VD4019"].astype(float).fillna(0.0).where(occupied, 0.0)
    activity_code = np.where(occupied, code("V4013", 5), "")
    persons = pd.DataFrame({
        "household": p["household"].to_numpy(), "status": status,
        "activity_code": activity_code,
        "earnings": earnings.to_numpy(),
        "education": pd.to_numeric(p["VD3005"], errors="coerce").fillna(0).to_numpy(),
        "female": (code("V2007") == "2").astype(float),
        "age": age.to_numpy()})
    if activities is not None:
        codes = pd.Series(activity_code)
        mapped = _by_prefix(codes, dict(activities)).where(pd.Series(occupied), None)
        lost = codes[pd.Series(occupied) & mapped.isna()]
        if len(lost):
            raise ValueError(
                f"{len(lost)} occupied persons have activity codes no prefix of `activities` "
                f"maps, such as {sorted(set(lost))[:5]}; map them, or the split of their "
                f"earnings among activities would have no rule")
        persons["activity"] = mapped.to_numpy()

    weight = p.assign(ref=code("V2005", 2) == "01")
    weight = weight.sort_values("ref", ascending=False).groupby("household")["V1028"].first()
    hh = pd.DataFrame({"weight": weight})
    hh["size"] = persons.groupby("household").size()
    for key in ("formal", "informal"):
        seg = segments[key]
        hh[key] = persons[persons["status"] == seg].groupby("household")["earnings"].sum()
    hh = hh.fillna({"formal": 0.0, "informal": 0.0}).reset_index()
    hh["group"] = hh[group] if isinstance(group, str) else np.asarray(group(hh))
    survey = Survey(hh.set_index("household", drop=False),
                    sources={"formal": segments["formal"], "informal": segments["informal"]})
    return survey, PersonSurvey(persons, characteristics=["education", "female", "age"])


# ---------------------------------------------------------------------------
# POF 2017–2018
# ---------------------------------------------------------------------------

#: Quadros whose values are monthly, so that V9011, the number of months, enters
#: the annualisation, by record, as in the IBGE's tabulation programs.
POF_MONTHLY_QUADROS = {"aluguel_estimado": None,            # every row
                       "despesa_coletiva": (10, 19),
                       "despesa_individual": (44, 47, 48, 49, 50),
                       "caderneta_coletiva": (),
                       "rendimento_trabalho": None,
                       "outros_rendimentos": (54,)}
_EXPENSES = ("aluguel_estimado", "despesa_coletiva", "caderneta_coletiva", "despesa_individual")


def _monthly(frame, record, value):
    """Monthly value per consumption unit: deflated, annualised, over twelve."""
    quadros = POF_MONTHLY_QUADROS[record]
    months = frame["V9011"] if "V9011" in frame else pd.Series(1.0, index=frame.index)
    if quadros is None:
        use = pd.Series(True, index=frame.index)
    else:
        use = frame["QUADRO"].isin(quadros)
    factor = np.where(use, months, 1.0)
    return frame[value] * factor * frame["FATOR_ANUALIZACAO"] / 12.0


def _code5(v9001):
    """The five-digit item code of the IBGE's translators: ``trunc(V9001 / 100)``."""
    return (pd.to_numeric(v9001, errors="coerce") // 100).astype("Int64").astype(str).str.zfill(5)


def _by_prefix(codes: pd.Series, table: Mapping[str, str]) -> pd.Series:
    """Map each code to the value of its longest matching prefix in ``table``."""
    out = pd.Series(None, index=codes.index, dtype=object)
    for prefix in sorted(table, key=len):
        out = out.where(~codes.str.startswith(str(prefix)), table[prefix])
    return out


def pof(directory, program, *, commodities: Mapping[str, str], group,
        income: Mapping[str, str] | None = None, exclude_conditions: Sequence[int] = (18, 19),
        records: Mapping[str, pd.DataFrame] | None = None):
    """The POF 2017–2018 as a household ``Survey`` of consumption units.

    Spending comes from the four records of the IBGE's general spending table
    (estimated rent, collective spending, the collective notebook and
    individual spending), each value ``V8000_DEFLA × FATOR_ANUALIZACAO / 12``,
    times ``V9011`` (months) for the quadros the IBGE treats as monthly:
    monthly at January 2018 prices. ``commodities`` maps a prefix of the
    five-digit item code to a SAM commodity; the longest prefix wins, and what
    no prefix maps is kept in ``unmapped_spending``, outside the budget.

    Labour income is the work-income record (``labour``); other income
    (``V8500_DEFLA``, V9011 for quadro 54) goes to the column ``income`` maps
    its code prefix to, or to ``other_income``. The household size counts the
    residents of the consumption unit except ``exclude_conditions`` of V0306
    (18 and 19: domestic employees and their relatives). The weight is
    PESO_FINAL. ``group`` assigns each unit to a SAM account, as in
    :func:`pnadc`. The returned survey maps ``labour`` to ``"fixed"`` and every
    income column likewise: replace its ``sources`` with the model's.
    ``records`` passes the records already read, by data-step name.
    """
    from .micro import Survey

    keys = ["COD_UPA", "NUM_DOM", "NUM_UC"]

    def load(name, variables):
        if records is not None and name in records:
            return records[name]
        layouts = read_sas_program(program)
        if name not in layouts:
            raise KeyError(f"record {name!r} is not in the POF program")
        path = os.path.join(directory, layouts[name].infile)
        if not os.path.exists(path):
            alt = [f for f in os.listdir(directory) if f.lower() == layouts[name].infile.lower()]
            if not alt:
                raise FileNotFoundError(f"{layouts[name].infile} is not in {directory}")
            path = os.path.join(directory, alt[0])
        return read_ibge(path, program, record=name, variables=variables)

    def uc(frame):
        return (frame["COD_UPA"].astype("Int64").astype(str) + "-"
                + frame["NUM_DOM"].astype("Int64").astype(str) + "-"
                + frame["NUM_UC"].astype("Int64").astype(str))

    people = load("morador", keys + ["V0306", "PESO_FINAL"])
    people["uc"] = uc(people)
    counted = ~pd.to_numeric(people["V0306"], errors="coerce").isin(list(exclude_conditions))
    hh = pd.DataFrame({"weight": people.groupby("uc")["PESO_FINAL"].first(),
                       "size": counted.groupby(people["uc"]).sum().astype(float)})
    if (hh["size"] <= 0).any():
        raise ValueError(f"{int((hh['size'] <= 0).sum())} consumption units have no resident "
                         f"left after excluding conditions {list(exclude_conditions)}")

    spend = []
    for name in _EXPENSES:
        variables = keys + ["QUADRO", "V9001", "V8000_DEFLA", "FATOR_ANUALIZACAO"]
        if POF_MONTHLY_QUADROS[name] != ():
            variables.append("V9011")
        r = load(name, variables)
        r = r[r["V8000_DEFLA"].notna()]
        spend.append(pd.DataFrame({"uc": uc(r), "code": _code5(r["V9001"]),
                                   "value": _monthly(r, name, "V8000_DEFLA")}))
    spend = pd.concat(spend, ignore_index=True)
    spend["commodity"] = _by_prefix(spend["code"], commodities)
    mapped = spend.dropna(subset=["commodity"])
    table = mapped.pivot_table(index="uc", columns="commodity", values="value", aggfunc="sum")
    budget = {}
    for c in table.columns:
        col = f"spend_{c}"
        hh[col] = table[c]
        budget[col] = c
    hh["unmapped_spending"] = spend[spend["commodity"].isna()].groupby("uc")["value"].sum()

    work = load("rendimento_trabalho", keys + ["QUADRO", "V9001", "V8500_DEFLA", "V9011",
                                               "FATOR_ANUALIZACAO"])
    work = work[work["V8500_DEFLA"].notna()]
    hh["labour"] = _monthly(work, "rendimento_trabalho", "V8500_DEFLA").groupby(uc(work)).sum()
    other = load("outros_rendimentos", keys + ["QUADRO", "V9001", "V8500_DEFLA", "V9011",
                                               "FATOR_ANUALIZACAO"])
    other = other[other["V8500_DEFLA"].notna()]
    o = pd.DataFrame({"uc": uc(other), "code": _code5(other["V9001"]),
                      "value": _monthly(other, "outros_rendimentos", "V8500_DEFLA")})
    o["column"] = _by_prefix(o["code"], dict(income or {})).fillna("other_income")
    for col, part in o.groupby("column"):
        hh[col] = part.groupby("uc")["value"].sum()
    hh = hh.fillna(0.0)
    hh.index.name = "household"
    hh = hh.reset_index()
    hh["group"] = hh[group] if isinstance(group, str) else np.asarray(group(hh))
    income_cols = ["labour"] + sorted(set(o["column"]))
    return Survey(hh.set_index("household", drop=False),
                  sources={c: "fixed" for c in income_cols if c in hh.columns},
                  budget=budget)
