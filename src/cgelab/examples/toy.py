"""A small SAM that balances exactly, built by hand.

Two activities, two commodities, two factors, two households, plus government,
investment and the rest of the world. Every entry is written out, so the base
year is known on paper and the base-year reproduction test compares the solver
against arithmetic rather than against its own previous output.

Three variants, because three code paths need exercising and each is a place
where a real SAM breaks a naive calibration:

* :func:`build` -- the plain case, no margins.
* :func:`build_with_margins` -- a trade-margin commodity that sells services into
  every other commodity's supply column against a negative entry on its own
  diagonal, which is the national-accounts convention.
* :func:`build_with_own_production` -- an activity that sells DIRECTLY to
  households and has no market share in any commodity: self-generation, home
  production, own-account work.
* :func:`build_with_trade_taxes` -- part of the product tax moved onto a tariff
  account and an indirect tax on output, the TRF and IDT of Hosoe's ``stdcge``.
"""

from __future__ import annotations

import pandas as pd

from ..calibrate import calibrate
from ..sam import SAM
from ..spec import ModelSpec

__all__ = ["build", "build_with_margins", "build_with_own_production", "build_with_twin_accounts",
           "build_with_institutions", "build_with_specific_capital", "build_with_trade_taxes",
           "spec"]


def _frame(entries) -> SAM:
    df = pd.DataFrame(entries, columns=["row", "col", "value"])
    return SAM.from_long(df, name="toy")


def spec(**overrides) -> ModelSpec:
    kwargs = dict(activity_prefix="a_", commodity_prefix="c_", household_prefix="hh_",
                  factors=("f_lab", "f_cap"), government="gov", investment="s_i",
                  rest_of_world="row", sigma_va=0.8, sigma_armington=2.0)
    kwargs.update(overrides)
    return ModelSpec(**kwargs)


_BASE = [
    # intermediates: commodity -> activity, at purchaser prices
    ("c_agr", "a_agr", 20.0), ("c_ind", "a_agr", 10.0),
    ("c_agr", "a_ind", 30.0), ("c_ind", "a_ind", 60.0),
    # value added: factor -> activity
    ("f_lab", "a_agr", 40.0), ("f_cap", "a_agr", 25.0),
    ("f_lab", "a_ind", 70.0), ("f_cap", "a_ind", 35.0),
    # production taxes on activities
    ("gov", "a_agr", 5.0), ("gov", "a_ind", 5.0),
    # make: activity -> commodity
    ("a_agr", "c_agr", 100.0), ("a_ind", "c_ind", 200.0),
    # imports and product taxes fill out the commodity supply columns
    ("row", "c_agr", 15.0), ("row", "c_ind", 25.0),
    ("gov", "c_agr", 5.0), ("gov", "c_ind", 10.0),
    # factor income to households
    ("hh_1", "f_lab", 70.0), ("hh_2", "f_lab", 40.0),
    ("hh_1", "f_cap", 45.0), ("hh_2", "f_cap", 15.0),
    # household consumption and saving
    ("c_agr", "hh_1", 30.0), ("c_ind", "hh_1", 55.0), ("s_i", "hh_1", 30.0),
    ("c_agr", "hh_2", 20.0), ("c_ind", "hh_2", 30.0), ("s_i", "hh_2", 5.0),
    # government demand and saving
    ("c_agr", "gov", 5.0), ("c_ind", "gov", 15.0), ("s_i", "gov", 5.0),
    # investment demand
    ("c_agr", "s_i", 5.0), ("c_ind", "s_i", 45.0),
    # exports and foreign saving
    ("c_agr", "row", 10.0), ("c_ind", "row", 20.0), ("s_i", "row", 10.0),
]


def build(model_spec: ModelSpec | None = None, **kwargs):
    """Return ``(sam, economy)`` for the plain two-sector case."""
    sam = _frame(_BASE)
    econ = calibrate(sam, model_spec or spec(), **kwargs)
    return sam, econ


def build_with_margins(model_spec: ModelSpec | None = None, **kwargs):
    """Same economy with a trade-margin commodity.

    ``c_trd`` sells 12 of margin services into the supply of the other two
    commodities and carries a NEGATIVE 12 on its own diagonal -- the offset that
    keeps its column balanced. Dropping the diagonal preserves the balance
    exactly, which is why the calibration is allowed to zero it; failing to read
    the off-diagonal block at all would lose the whole 12.
    """
    entries = [e for e in _BASE if not (e[0] == "a_ind" and e[1] == "c_ind")]
    entries += [
        ("a_trd", "c_trd", 12.0),
        ("c_trd", "c_agr", 5.0), ("c_trd", "c_ind", 7.0),
        ("c_trd", "c_trd", -12.0),
        ("f_lab", "a_trd", 8.0), ("f_cap", "a_trd", 4.0),
        ("a_ind", "c_ind", 200.0),
        # the margin services raise the supply value of c_agr and c_ind, so the
        # extra 12 has to be absorbed: households buy that much more.
        ("c_agr", "hh_1", 5.0), ("c_ind", "hh_1", 7.0),
        ("hh_1", "f_lab", 8.0), ("hh_1", "f_cap", 4.0),
    ]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(), **kwargs)
    return sam, econ


def build_with_own_production(model_spec: ModelSpec | None = None, **kwargs):
    """Same economy plus ``a_own``, an activity selling straight to households.

    It has no market share in any commodity, which is what makes its level
    something a scenario may set exogenously. ``a_own`` uses 9 of capital and
    sells 9 to households; that capital is an addition to the endowment, so
    household capital income rises by the same 9 and their commodity purchases
    are untouched. Cutting their commodity spending as well would have credited
    the income twice and the SAM would not balance -- which is exactly what the
    balance check is for.
    """
    drop = {("hh_1", "f_cap", 45.0), ("hh_2", "f_cap", 15.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [
        ("hh_1", "f_cap", 51.0), ("hh_2", "f_cap", 18.0),
        ("f_cap", "a_own", 9.0),
        ("a_own", "hh_1", 6.0), ("a_own", "hh_2", 3.0),
    ]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(), **kwargs)
    return sam, econ


def build_with_twin_accounts(model_spec: ModelSpec | None = None, **kwargs):
    """``hh_2`` split into two accounts with IDENTICAL preferences and saving rate.

    The pair stands for one population -- one decile, one region -- observed in
    two states between which households migrate: adopter and non-adopter, formal
    and informal. Because the twins are identical, moving income between them
    changes no aggregate and no price, so any welfare difference a measure
    reports for that move is composition and not welfare. That is what makes this
    the right fixture for :func:`cgelab.welfare.ev_by_group`.
    """
    drop = {("hh_2", "f_lab", 40.0), ("hh_2", "f_cap", 15.0),
            ("c_agr", "hh_2", 20.0), ("c_ind", "hh_2", 30.0), ("s_i", "hh_2", 5.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    for tag in ("2a", "2b"):
        entries += [(f"hh_{tag}", "f_lab", 20.0), (f"hh_{tag}", "f_cap", 7.5),
                    ("c_agr", f"hh_{tag}", 10.0), ("c_ind", f"hh_{tag}", 15.0),
                    ("s_i", f"hh_{tag}", 2.5)]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(), **kwargs)
    return sam, econ


def build_with_institutions(model_spec: ModelSpec | None = None, **kwargs):
    """Same economy with the flows between institutions that a real SAM carries.

    * capital income of 60 goes 35 to ``hh_1``, 11 to ``hh_2``, 4 to government
      (public enterprises) and 10 abroad (foreign-owned capital);
    * direct taxes of 12 (``hh_1``) and 2 (``hh_2``);
    * a government transfer of 8 to ``hh_2``;
    * remittances of 3 from abroad to ``hh_2`` and of 1 from ``hh_1`` abroad.

    Savings close each account: ``hh_1`` 105 - 85 - 12 - 1 = 7, ``hh_2``
    62 - 50 - 2 = 10, government 43 - 20 - 8 = 15, and foreign saving is the
    current account, 40 + 10 + 1 - 30 - 3 = 18. They sum to the investment of 50.
    """
    drop = {("hh_1", "f_cap", 45.0), ("hh_2", "f_cap", 15.0), ("s_i", "hh_1", 30.0),
            ("s_i", "hh_2", 5.0), ("s_i", "gov", 5.0), ("s_i", "row", 10.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [
        ("hh_1", "f_cap", 35.0), ("hh_2", "f_cap", 11.0),
        ("gov", "f_cap", 4.0), ("row", "f_cap", 10.0),
        ("gov", "hh_1", 12.0), ("gov", "hh_2", 2.0),
        ("hh_2", "gov", 8.0),
        ("hh_2", "row", 3.0), ("row", "hh_1", 1.0),
        ("s_i", "hh_1", 7.0), ("s_i", "hh_2", 10.0), ("s_i", "gov", 15.0), ("s_i", "row", 18.0),
    ]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(), **kwargs)
    return sam, econ


def build_with_specific_capital(model_spec: ModelSpec | None = None, **kwargs):
    """Same economy with the industrial capital a factor of its own, ``f_kind``.

    ``a_ind`` uses 35 of ``f_kind`` instead of mobile ``f_cap``, the way a
    sector-specific stock (a refinery, an electrolyser park) is represented.
    Household capital income is unchanged in total: ``hh_1`` 19 + 26, ``hh_2`` 6 + 9.
    """
    drop = {("f_cap", "a_ind", 35.0), ("hh_1", "f_cap", 45.0), ("hh_2", "f_cap", 15.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [
        ("f_kind", "a_ind", 35.0),
        ("hh_1", "f_cap", 19.0), ("hh_2", "f_cap", 6.0),
        ("hh_1", "f_kind", 26.0), ("hh_2", "f_kind", 9.0),
    ]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(factors=("f_lab", "f_cap", "f_kind")), **kwargs)
    return sam, econ


def build_with_trade_taxes(model_spec: ModelSpec | None = None, **kwargs):
    """The plain economy with a tariff account and an indirect tax on output.

    Of the product tax on agriculture (5) and industry (10), 1 and 2 become
    tariff on imports of 15 and 25, and 2 and 4 become indirect tax on output of
    100 and 200. Government receives both accounts whole. The purchaser value
    of every commodity is the plain economy's.
    """
    moved = {"c_agr": (1.0, 2.0), "c_ind": (2.0, 4.0)}
    entries = []
    for row, col, value in _BASE:
        if row == "gov" and col in moved:
            trf, idt = moved[col]
            entries += [("gov", col, value - trf - idt), ("trf", col, trf), ("idt", col, idt)]
        else:
            entries.append((row, col, value))
    entries += [("gov", "trf", 3.0), ("gov", "idt", 6.0)]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(tariff_account="trf", indirect_tax_account="idt"),
                     **kwargs)
    return sam, econ
