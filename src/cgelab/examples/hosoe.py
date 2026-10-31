"""The standard CGE model of Hosoe, Gasawa and Hashimoto (2004), chapter 6.

The SAM is the one printed in the GAMS model library files ``stdcge`` and
``lrgcge``. Hosoe's accounts BRD and MLK are an activity and a commodity at
once; here each is split into ``a_`` and ``c_``, and the make matrix is the
diagonal of gross output (73 and 72). Everything else is the published table:
an indirect tax on output (IDT), a tariff (TRF), one household, government,
investment and the rest of the world.

:func:`build` returns the economy with Hosoe's functional forms: Cobb-Douglas
value added, Leontief intermediates, Armington and CET elasticities of 2.
:data:`CLOSURE` is his closure: government saves a fixed share of its income
and spends the rest in fixed value shares, and foreign saving is fixed in
dollars. ``large_country=True`` adds the export-demand and import-supply curves
of ``lrgcge``, with elasticities of 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibrate import calibrate
from ..closure import Closure
from ..sam import SAM
from ..spec import ModelSpec

__all__ = ["ENTRIES", "CLOSURE", "spec", "build", "variables", "SIMPLE_ENTRIES", "build_simple"]

ENTRIES = [
    ("c_brd", "a_brd", 21.0), ("c_brd", "a_mlk", 8.0), ("c_brd", "hh", 20.0),
    ("c_brd", "gov", 19.0), ("c_brd", "s_i", 16.0), ("c_brd", "row", 8.0),
    ("c_mlk", "a_brd", 17.0), ("c_mlk", "a_mlk", 9.0), ("c_mlk", "hh", 30.0),
    ("c_mlk", "gov", 14.0), ("c_mlk", "s_i", 15.0), ("c_mlk", "row", 4.0),
    ("f_cap", "a_brd", 20.0), ("f_cap", "a_mlk", 30.0),
    ("f_lab", "a_brd", 15.0), ("f_lab", "a_mlk", 25.0),
    ("a_brd", "c_brd", 73.0), ("a_mlk", "c_mlk", 72.0),
    ("idt", "c_brd", 5.0), ("idt", "c_mlk", 4.0),
    ("trf", "c_brd", 1.0), ("trf", "c_mlk", 2.0),
    ("row", "c_brd", 13.0), ("row", "c_mlk", 11.0),
    ("hh", "f_cap", 50.0), ("hh", "f_lab", 40.0),
    ("gov", "idt", 9.0), ("gov", "trf", 3.0), ("gov", "hh", 23.0),
    ("s_i", "hh", 17.0), ("s_i", "gov", 2.0), ("s_i", "row", 12.0),
]

CLOSURE = Closure(government="saving_rate", foreign_saving_unit="foreign_currency")


def spec(large_country: bool = False, **overrides) -> ModelSpec:
    kwargs = dict(
        activities=["a_brd", "a_mlk"], commodities=["c_brd", "c_mlk"], households=["hh"],
        factors=("f_cap", "f_lab"), numeraire_factor="f_lab",
        government="gov", investment="s_i", rest_of_world="row",
        value_added="cobb_douglas", sigma_armington=2.0,
        export_rule="cet", sigma_cet=2.0, margins=False,
        tariff_account="trf", indirect_tax_account="idt")
    if large_country:
        kwargs.update(export_demand=2.0, import_supply=2.0)
    kwargs.update(overrides)
    return ModelSpec(**kwargs)


def build(large_country: bool = False, model_spec: ModelSpec | None = None):
    """Return ``(sam, economy)`` for ``stdcge``, or ``lrgcge`` when ``large_country``."""
    sam = SAM.from_long(pd.DataFrame(ENTRIES, columns=["row", "col", "value"]),
                        name="hosoe_stdcge")
    return sam, calibrate(sam, model_spec or spec(large_country))


#: Chapter 5, the GAMS library file ``splcge``: bread and milk, capital and
#: labour, one household, no intermediates, no taxes and no trade. The package
#: needs government, investment and rest-of-world accounts, so they are empty.
SIMPLE_ENTRIES = [
    ("c_brd", "hh_1", 15.0), ("c_mlk", "hh_1", 35.0),
    ("f_cap", "a_brd", 5.0), ("f_cap", "a_mlk", 20.0),
    ("f_lab", "a_brd", 10.0), ("f_lab", "a_mlk", 15.0),
    ("hh_1", "f_cap", 25.0), ("hh_1", "f_lab", 25.0),
    ("a_brd", "c_brd", 15.0), ("a_mlk", "c_mlk", 35.0),
    ("gov", "gov", 0.0), ("s_i", "s_i", 0.0), ("row", "row", 0.0),
]


def build_simple():
    """Return ``(sam, economy)`` for ``splcge``, Cobb-Douglas with labour as numeraire."""
    sam = SAM.from_long(pd.DataFrame(SIMPLE_ENTRIES, columns=["row", "col", "value"]),
                        name="hosoe_splcge")
    spec_ = ModelSpec(
        activities=["a_brd", "a_mlk"], commodities=["c_brd", "c_mlk"],
        households=["hh_1"], factors=("f_lab", "f_cap"),
        government="gov", investment="s_i", rest_of_world="row",
        sigma_va=1.0, margins=False, numeraire_factor="f_lab")
    return sam, calibrate(sam, spec_)


def variables(econ, res) -> dict:
    """Hosoe's variables, under his names, from a solved period.

    Vectors are keyed ``BRD``/``MLK`` (and ``CAP``/``LAB`` for factors), the
    way the GAMS listing prints them, so the result can be set against the
    published program's own output.
    """
    from ..solve import value_added_price

    goods, facs = ("BRD", "MLK"), ("CAP", "LAB")

    def by_good(v):
        return {g: float(x) for g, x in zip(goods, v)}

    tm = econ.tm0 if res.scenario.tariff is None else np.asarray(res.scenario.tariff, float)
    tauz = (econ.tauz0 if res.scenario.output_tax is None
            else np.asarray(res.scenario.output_tax, float))
    pm = res.exchange_rate * res.world_import_price
    hh_sav = float(((res.hh_income - res.hh_direct_tax) * econ.hh_savrate).sum())
    xp = res.hh_quantity.sum(axis=0)
    alpha = econ.hh_budget[0]
    fac = {f.removeprefix("f_").upper(): i for i, f in enumerate(econ.factors)}
    out = {
        "Y": by_good(econ.vunit * res.X),
        "F": {f"{h},{g}": float(res.factor_demand_by_activity[fac[h], j])
              for h in facs for j, g in enumerate(goods)},
        "Z": by_good(res.Q), "Xp": by_good(xp), "Xg": by_good(res.gov_quantity),
        "Xv": by_good(res.investment_quantity), "E": by_good(res.exports),
        "M": by_good(res.imports), "Q": by_good(res.absorption),
        "D": by_good(res.domestic_sales),
        "pf": {h: float(res.factor_prices[fac[h]]) for h in facs},
        "py": by_good(value_added_price(econ, res.factor_prices)),
        "pz": by_good(res.pd), "pq": by_good(res.pq), "pe": by_good(res.pe),
        "pm": by_good(pm), "pd": by_good(res.p_domestic),
        "pWe": by_good(res.world_export_price), "pWm": by_good(res.world_import_price),
        "epsilon": float(res.exchange_rate),
        "Sp": hh_sav, "Sg": float(res.savings - hh_sav - res.foreign_saving),
        "Td": float(res.hh_direct_tax.sum()),
        "Tz": by_good(tauz * res.pd * res.Q), "Tm": by_good(tm * pm * res.imports),
        "UU": float(np.prod(xp ** alpha)),
    }
    if res.scenario.import_quota:
        # quocge's quasi-rent chi and its rent RT, zero on a good without a quota.
        quota = res.quota or {}
        out["chi"] = by_good([quota.get(c, {}).get("premium", 0.0) for c in econ.coms])
        out["RT"] = by_good([quota.get(c, {}).get("rent", 0.0) for c in econ.coms])
    return out
