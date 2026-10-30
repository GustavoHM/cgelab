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
* :func:`build_with_segmented_labour` -- labour split into a formal and an
  informal account that share one population of workers.
* :func:`build_with_technologies` -- industry made by two technologies, one
  capital-intensive and one labour-intensive, that compete at least cost.
"""

from __future__ import annotations

import pandas as pd

from ..calibrate import calibrate
from ..sam import SAM
from ..spec import ModelSpec

__all__ = ["build", "build_with_margins", "build_with_own_production", "build_with_twin_accounts",
           "build_with_institutions", "build_with_specific_capital", "build_with_trade_taxes",
           "build_with_segmented_labour", "build_with_technologies", "survey", "persons",
           "build_one_good", "spec"]


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


#: The industrial activity split into two technologies for ``c_ind``: ``a_ind``
#: capital-intensive (120) and ``a_alt`` labour-intensive (80). Their columns
#: add up to the original activity's, so the rest of the SAM is unchanged.
TECHNOLOGIES = [
    ("c_agr", "a_ind", 18.0), ("c_ind", "a_ind", 36.0),
    ("f_lab", "a_ind", 30.0), ("f_cap", "a_ind", 33.0), ("gov", "a_ind", 3.0),
    ("a_ind", "c_ind", 120.0),
    ("c_agr", "a_alt", 12.0), ("c_ind", "a_alt", 24.0),
    ("f_lab", "a_alt", 40.0), ("f_cap", "a_alt", 2.0), ("gov", "a_alt", 2.0),
    ("a_alt", "c_ind", 80.0),
]


def build_with_technologies(model_spec: ModelSpec | None = None, **kwargs):
    """Industry made by two technologies that compete at least cost.

    ``ModelSpec.least_cost = ["c_ind"]`` unless ``model_spec`` says otherwise:
    under a shock that makes one technology cheaper the other shuts down.
    """
    entries = [e for e in _BASE if "a_ind" not in (e[0], e[1])] + TECHNOLOGIES
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(least_cost=["c_ind"]), **kwargs)
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


def build_with_segmented_labour(model_spec: ModelSpec | None = None, **kwargs):
    """The plain economy with labour split into a formal and an informal account.

    Industry pays its 70 of labour to ``f_for``, agriculture its 40 to
    ``f_inf``, and the two form one group of segments, ``"labour"``: the same
    people supply both. Formal labour is owned mostly by ``hh_1`` (60 of 70),
    informal mostly by ``hh_2`` (30 of 40), so who loses a formal job matters
    for the distribution. Capital is the numeraire.
    """
    drop = {("f_lab", "a_agr", 40.0), ("f_lab", "a_ind", 70.0),
            ("hh_1", "f_lab", 70.0), ("hh_2", "f_lab", 40.0)}
    entries = [e for e in _BASE if tuple(e) not in drop]
    entries += [("f_inf", "a_agr", 40.0), ("f_for", "a_ind", 70.0),
                ("hh_1", "f_inf", 10.0), ("hh_2", "f_inf", 30.0),
                ("hh_1", "f_for", 60.0), ("hh_2", "f_for", 10.0)]
    sam = _frame(entries)
    econ = calibrate(sam, model_spec or spec(
        factors=("f_cap", "f_for", "f_inf"), numeraire_factor="f_cap",
        labour_segments={"labour": ["f_for", "f_inf"]}), **kwargs)
    return sam, econ


def survey():
    """A synthetic household survey of :func:`build_with_segmented_labour`.

    Twelve households, six in each SAM account, written by hand: survey weights,
    sizes, formal and informal wages, capital income, and spending on food and
    goods. It is not data. Its weighted totals deliberately differ from the
    SAM's (formal wages 51 against 60 in ``hh_1``, capital 8.75 against 15 in
    ``hh_2``), as a real survey's do, so that reconciliation has work to do.
    """
    from ..micro import Survey
    rows = [
        # group, weight, size, formal, informal, capital, food, goods
        ("hh_1", 2.0, 4, 8.0, 0.0, 2.0, 3.0, 5.0),
        ("hh_1", 1.5, 3, 10.0, 0.0, 4.0, 3.0, 6.0),
        ("hh_1", 1.0, 2, 12.0, 0.0, 9.0, 3.0, 8.0),
        ("hh_1", 1.0, 5, 0.0, 4.0, 1.0, 4.0, 2.0),
        ("hh_1", 1.0, 1, 6.0, 1.0, 6.0, 2.0, 5.0),
        ("hh_1", 0.5, 3, 4.0, 3.0, 12.0, 3.0, 9.0),
        ("hh_2", 3.0, 5, 0.0, 3.0, 0.0, 2.0, 1.0),
        ("hh_2", 2.5, 4, 2.0, 2.0, 0.5, 2.0, 1.0),
        ("hh_2", 2.0, 6, 0.0, 4.0, 0.0, 3.0, 1.0),
        ("hh_2", 1.5, 3, 3.0, 1.0, 1.0, 2.0, 2.0),
        ("hh_2", 1.0, 2, 0.0, 2.0, 2.0, 1.0, 2.0),
        ("hh_2", 1.0, 4, 1.0, 5.0, 4.0, 3.0, 3.0),
    ]
    frame = pd.DataFrame(rows, columns=["group", "weight", "size", "formal", "informal",
                                        "capital", "food", "goods"])
    return Survey(frame, sources={"formal": "f_for", "informal": "f_inf", "capital": "f_cap"},
                  budget={"food": "c_agr", "goods": "c_ind"})


def persons():
    """The 42 persons of the twelve households of :func:`survey`, synthetic too.

    Each has a status (formal ``f_for``, informal ``f_inf``, ``"unemployed"``
    or ``"inactive"``, which includes children), labour earnings, years of
    education, a ``female`` dummy and an age. The earnings need not add up to
    the household survey's wage columns: each survey is reconciled with the SAM
    on its own.
    """
    from ..behaviour import PersonSurvey
    people = {
        0: [("f_for", 8.0, 11, 0, 40), ("inactive", 0, 8, 1, 38), ("inactive", 0, 3, 1, 10),
            ("inactive", 0, 1, 0, 6)],
        1: [("f_for", 6.0, 12, 0, 35), ("f_for", 4.0, 9, 1, 33), ("unemployed", 0, 9, 0, 19)],
        2: [("f_for", 12.0, 15, 1, 45), ("inactive", 0, 12, 0, 70)],
        3: [("f_inf", 3.0, 6, 0, 42), ("f_inf", 1.0, 5, 1, 40), ("unemployed", 0, 7, 0, 20),
            ("inactive", 0, 2, 1, 9), ("inactive", 0, 0, 0, 4)],
        4: [("f_for", 7.0, 13, 1, 29)],
        5: [("f_for", 4.0, 10, 0, 50), ("f_inf", 3.0, 8, 1, 48), ("inactive", 0, 9, 0, 75)],
        6: [("f_inf", 2.0, 4, 0, 36), ("f_inf", 1.0, 3, 1, 34), ("unemployed", 0, 6, 0, 18),
            ("inactive", 0, 1, 1, 8), ("inactive", 0, 0, 0, 3)],
        7: [("f_for", 2.0, 9, 1, 30), ("f_inf", 2.0, 6, 0, 32), ("inactive", 0, 4, 1, 7),
            ("inactive", 0, 1, 0, 2)],
        8: [("f_inf", 2.0, 5, 0, 45), ("f_inf", 1.5, 4, 1, 43), ("f_inf", 0.5, 7, 0, 17),
            ("unemployed", 0, 8, 1, 21), ("inactive", 0, 2, 0, 9), ("inactive", 0, 0, 1, 5)],
        9: [("f_for", 3.0, 11, 0, 39), ("f_inf", 1.0, 7, 1, 37), ("inactive", 0, 5, 0, 12)],
        10: [("f_inf", 2.0, 8, 1, 52), ("unemployed", 0, 6, 0, 24)],
        11: [("f_for", 1.0, 10, 0, 28), ("f_inf", 3.0, 9, 1, 47), ("f_inf", 2.0, 7, 0, 49),
             ("unemployed", 0, 11, 1, 22)],
    }
    rows = [(h, *p) for h, members in people.items() for p in members]
    frame = pd.DataFrame(rows, columns=["household", "status", "earnings", "education",
                                        "female", "age"])
    frame["earnings"] = frame["earnings"].astype(float)
    return PersonSurvey(frame, characteristics=["education", "female"],
                        choice_characteristics=["education"])


def build_one_good():
    """One good, two factors, one household, closed: the stochastic growth model.

    Labour 60 and capital 40 make 100 of the good, Cobb-Douglas; households
    consume 80 and invest 20. With a user cost of 0.10 the capital stock is 400,
    and 20 is exactly its depreciation at 5%, so the base year is a steady state
    of :func:`cgelab.bridge.ramsey` at ``discount=depreciation=0.05``.
    """
    entries = [
        ("f_lab", "a_x", 60.0), ("f_cap", "a_x", 40.0), ("a_x", "c_x", 100.0),
        ("c_x", "hh", 80.0), ("c_x", "s_i", 20.0),
        ("hh", "f_lab", 60.0), ("hh", "f_cap", 40.0), ("s_i", "hh", 20.0),
        ("gov", "gov", 0.0), ("row", "row", 0.0),
    ]
    sam = _frame(entries)
    spec_ = ModelSpec(activities=["a_x"], commodities=["c_x"], households=["hh"],
                      factors=("f_lab", "f_cap"), government="gov", investment="s_i",
                      rest_of_world="row", sigma_va=1.0, margins=False, numeraire_factor="f_lab")
    return sam, calibrate(sam, spec_)
