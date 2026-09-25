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
"""

from __future__ import annotations

import pandas as pd

from ..calibrate import calibrate
from ..sam import SAM
from ..spec import ModelSpec

__all__ = ["build", "build_with_margins", "build_with_own_production", "spec"]


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
    the right fixture for :func:`cgekit.welfare.ev_by_group`.
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
