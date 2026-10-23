"""Building a SAM from supply and use tables, with the source of every block.

Most of the work in an applied model happens before ``calibrate``: a SAM is
assembled from the national accounts. The supply and use tables give the
production block (who makes what, who uses what, at which prices, with which
taxes and margins) and final demand. They do not give the distribution of
income among institutions: that comes from the integrated economic accounts,
or is assumed. :func:`sut_to_sam` assembles both in the layout ``calibrate``
reads, and records which source each block came from, because a SAM whose
provenance is lost cannot be revised when one of its sources is.

The layout is the package's: activities pay intermediate inputs at purchaser
prices and their factors; a commodity's column is its supply, made by
activities at basic prices, imported, taxed and carried, with trade and
transport margins in the commodity block (the margin commodity selling into
the others against a negative entry on its own diagonal); savings close the
institutions and fund investment.

The savings of households, government and the rest of the world are residuals,
declared as such. The investment account then balances only if the tables do,
and a gap there is reported, not closed: it is evidence about the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sam import SAM

__all__ = ["SupplyUse", "Institutions", "sut_to_sam", "split_households", "split_factor",
           "split_vintages", "segment_shares", "group_population", "read_tru"]

_ROLES = ("households", "government", "investment", "exports")


def _frame(x, name, rows=None, cols=None) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame(index=rows or [], columns=cols or [], dtype=float)
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"SupplyUse.{name} must be a labelled DataFrame")
    return x.astype(float).fillna(0.0)


def _series(x, name) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.DataFrame):          # several taxes on the same commodity
        return x.astype(float).fillna(0.0).sum(axis=1)
    if not isinstance(x, pd.Series):
        raise TypeError(f"SupplyUse.{name} must be a labelled Series or DataFrame")
    return x.astype(float).fillna(0.0)


@dataclass
class SupplyUse:
    """Supply and use tables, labelled by the accounts they become.

    * ``make`` -- activities by commodities, output at basic prices;
    * ``use`` -- commodities by activities, intermediate use at purchaser prices;
    * ``final`` -- commodities by final-demand columns, at purchaser prices.
      ``final_roles`` says what each column is: ``"households"`` (the column is
      a household account), ``"government"``, ``"investment"`` (fixed capital
      formation, changes in stocks: summed into the investment account) or
      ``"exports"``. A column named like the government, investment or
      rest-of-world account needs no role; any other is a household;
    * ``imports`` -- by commodity, at border prices;
    * ``value_added`` -- factors by activities;
    * ``product_taxes`` -- by commodity (a DataFrame of several taxes is summed);
    * ``production_taxes`` -- by activity, net of subsidies;
    * ``margins`` -- margin commodities by commodities: what each margin
      commodity adds to each commodity's supply, and minus its own total on
      its own column, so that each row adds up to zero;
    * ``tariffs``, ``output_taxes`` -- by commodity, when the SAM keeps them in
      accounts of their own (``tariff_account``, ``indirect_tax_account`` of
      ``ModelSpec``); otherwise they belong in ``product_taxes``;
    * ``direct_sales`` -- activities by household columns: own-account
      production sold straight to households, outside the commodity market.
    """

    make: pd.DataFrame
    use: pd.DataFrame
    final: pd.DataFrame
    imports: pd.Series
    value_added: pd.DataFrame
    product_taxes: pd.Series | pd.DataFrame | None = None
    production_taxes: pd.Series | None = None
    margins: pd.DataFrame | None = None
    tariffs: pd.Series | None = None
    output_taxes: pd.Series | None = None
    direct_sales: pd.DataFrame | None = None
    final_roles: Mapping[str, str] | None = None
    #: Descriptions of the accounts, ``{account: label}``, when a reader knows them.
    labels: Mapping[str, str] | None = None
    #: How the tables were put together, recorded in the SAM's provenance:
    #: ``{note: source}`` (an allocation the reader assumed, say).
    notes: Mapping[str, str] | None = None

    def __post_init__(self):
        self.make = _frame(self.make, "make")
        self.use = _frame(self.use, "use")
        self.final = _frame(self.final, "final")
        self.value_added = _frame(self.value_added, "value_added")
        self.imports = _series(self.imports, "imports")
        self.product_taxes = _series(self.product_taxes, "product_taxes")
        self.production_taxes = _series(self.production_taxes, "production_taxes")
        self.tariffs = _series(self.tariffs, "tariffs")
        self.output_taxes = _series(self.output_taxes, "output_taxes")
        self.margins = _frame(self.margins, "margins")
        self.direct_sales = _frame(self.direct_sales, "direct_sales")
        for role in (self.final_roles or {}).values():
            if role not in _ROLES:
                raise ValueError(f"a final-demand role must be one of {_ROLES}, got {role!r}")
        acts = list(self.make.index)
        coms = list(self.make.columns)
        checks = [("use", list(self.use.index), coms, list(self.use.columns), acts),
                  ("final", list(self.final.index), coms, None, None),
                  ("value_added", None, None, list(self.value_added.columns), acts)]
        for name, rows, want_rows, cols, want_cols in checks:
            for got, want, axis in ((rows, want_rows, "rows"), (cols, want_cols, "columns")):
                if got is None:
                    continue
                stray = [x for x in got if x not in want]
                if stray:
                    raise KeyError(f"SupplyUse.{name} has {axis} {stray} that are not "
                                   f"{'commodities' if want is coms else 'activities'} of "
                                   f"make")
        for name in ("imports", "product_taxes", "tariffs", "output_taxes"):
            stray = [x for x in getattr(self, name).index if x not in coms]
            if stray:
                raise KeyError(f"SupplyUse.{name} has {stray}, which are not commodities")
        stray = [x for x in self.production_taxes.index if x not in acts]
        if stray:
            raise KeyError(f"SupplyUse.production_taxes has {stray}, which are not activities")
        stray = ([x for x in self.margins.index if x not in coms]
                 + [x for x in self.margins.columns if x not in coms])
        if stray:
            raise KeyError(f"SupplyUse.margins has {stray}, which are not commodities")

    @property
    def activities(self) -> list:
        return list(self.make.index)

    @property
    def commodities(self) -> list:
        return list(self.make.columns)

    def roles(self, names) -> dict:
        """The role of every final-demand column."""
        out = {}
        for col in self.final.columns:
            given = (self.final_roles or {}).get(col)
            if given is not None:
                out[col] = given
            elif col == names["government"]:
                out[col] = "government"
            elif col == names["investment"]:
                out[col] = "investment"
            elif col == names["rest_of_world"]:
                out[col] = "exports"
            else:
                out[col] = "households"
        return out

    def _vec(self, s, index):
        return s.reindex(index, fill_value=0.0).to_numpy()

    def supply(self) -> pd.Series:
        """Each commodity's supply at purchaser prices."""
        c = self.commodities
        margins = (self.margins.reindex(columns=c, fill_value=0.0).sum(axis=0).to_numpy()
                   if not self.margins.empty else 0.0)
        return pd.Series(self.make.sum(axis=0).reindex(c).to_numpy() + self._vec(self.imports, c)
                         + self._vec(self.product_taxes, c) + self._vec(self.tariffs, c)
                         + self._vec(self.output_taxes, c) + margins, index=c)

    def demand(self) -> pd.Series:
        """Each commodity's intermediate and final use at purchaser prices."""
        c = self.commodities
        return (self.use.reindex(index=c, fill_value=0.0).sum(axis=1)
                + self.final.reindex(index=c, fill_value=0.0).sum(axis=1))

    def output(self) -> pd.Series:
        """Each activity's output: its sales to commodities and straight to households."""
        a = self.activities
        direct = (self.direct_sales.reindex(index=a, fill_value=0.0).sum(axis=1)
                  if not self.direct_sales.empty else 0.0)
        return self.make.sum(axis=1) + direct

    def costs(self) -> pd.Series:
        """Each activity's intermediate use, value added and production taxes."""
        a = self.activities
        return (self.use.reindex(columns=a, fill_value=0.0).sum(axis=0)
                + self.value_added.reindex(columns=a, fill_value=0.0).sum(axis=0)
                + self.production_taxes.reindex(a, fill_value=0.0))

    def gdp(self, names=None) -> dict:
        """GDP from production, expenditure and income.

        Production is value added plus every tax on products and production;
        expenditure is final demand less imports, with exports in it; income
        is factor income plus the same taxes. They agree when the tables do.
        """
        names = {**_NAMES, **(names or {})}
        taxes = (float(self.product_taxes.sum()) + float(self.tariffs.sum())
                 + float(self.output_taxes.sum()) + float(self.production_taxes.sum()))
        direct = float(self.direct_sales.to_numpy().sum()) if not self.direct_sales.empty else 0.0
        production = float(self.output().sum() - self.use.to_numpy().sum()) + (
            taxes - float(self.production_taxes.sum()))
        expenditure = float(self.final.to_numpy().sum()) + direct - float(self.imports.sum())
        income = float(self.value_added.to_numpy().sum()) + taxes
        return dict(production=production, expenditure=expenditure, income=income)

    def check(self, tol: float = 1e-9) -> None:
        """Refuse tables that do not add up, naming where."""
        scale = max(float(np.abs(self.make.to_numpy()).sum()), 1.0)
        gap = self.supply() - self.demand()
        bad = gap[np.abs(gap) > tol * scale]
        if len(bad):
            raise ValueError(
                "supply and use disagree for commodities "
                + ", ".join(f"{k} (supply − use = {v:.6g})" for k, v in bad.items())
                + ". A SAM built on them would not balance; fix the tables, or say which "
                "entry is missing.")
        gap = self.output() - self.costs()
        bad = gap[np.abs(gap) > tol * scale]
        if len(bad):
            raise ValueError(
                "output and costs disagree for activities "
                + ", ".join(f"{k} (output − costs = {v:.6g})" for k, v in bad.items()))
        if not self.margins.empty:
            rows = self.margins.sum(axis=1)
            bad = rows[np.abs(rows) > tol * scale]
            if len(bad):
                raise ValueError(
                    "each margin row must add up to zero, the margin commodity's own column "
                    "carrying minus what it adds to the others; "
                    + ", ".join(f"{k} adds up to {v:.6g}" for k, v in bad.items()))


@dataclass
class Institutions:
    """What the supply and use tables do not say: who receives income, and transfers.

    * ``factor_income`` -- recipients by factors: household accounts, the
      government and the rest of the world. Each factor's column must add up
      to its value added. Absent, all of it goes to the single household
      account (refused when there are several);
    * ``direct_taxes`` -- by household, paid to government;
    * ``transfers`` -- by household, received from government;
    * ``from_abroad`` and ``to_abroad`` -- by household, transfers received
      from and paid to the rest of the world;
    * ``flows`` -- any other payment, ``(receiver, payer, amount, source)``,
      such as a grant from abroad to government.
    """

    factor_income: pd.DataFrame | None = None
    direct_taxes: pd.Series | None = None
    transfers: pd.Series | None = None
    from_abroad: pd.Series | None = None
    to_abroad: pd.Series | None = None
    flows: Sequence[tuple] = field(default_factory=list)
    #: The source each field is recorded under in the SAM's provenance.
    source: str = "integrated economic accounts"


_NAMES = dict(government="gov", investment="s_i", rest_of_world="row",
              tariff="trf", output_tax="idt")


def sut_to_sam(sut: SupplyUse, institutions: Institutions | None = None, *,
               names: Mapping[str, str] | None = None, source: str = "supply and use tables",
               name: str = "sam", tol: float = 1e-9) -> SAM:
    """Assemble a SAM from supply and use tables and the institutional accounts.

    Account names are the tables' labels; ``names`` overrides the government,
    investment, rest-of-world, tariff and output-tax accounts. The savings of
    each household, of government and of the rest of the world (foreign saving)
    are residuals. The investment account then balances exactly when the tables
    and the institutional accounts are consistent, and the gap is reported when
    they are not. ``SAM.provenance`` lists every block with its source.
    """
    names = {**_NAMES, **(names or {})}
    inst = institutions or Institutions()
    sut.check(tol)
    gov, inv, row = names["government"], names["investment"], names["rest_of_world"]
    roles = sut.roles(names)
    hhs = [c for c, r in roles.items() if r == "households"]
    if not sut.direct_sales.empty:
        stray = [c for c in sut.direct_sales.columns if c not in hhs]
        if stray:
            raise KeyError(f"SupplyUse.direct_sales goes to {stray}, which are not household "
                           f"columns of final demand")
        hhs += [c for c in sut.direct_sales.columns if c not in hhs]
    entries = []

    def add(block_rows, block_cols, frame, src, note=""):
        for r in frame.index:
            for c in frame.columns:
                v = float(frame.at[r, c])
                if v != 0.0:
                    entries.append((r, c, v, src, note))

    def add_vec(receiver, payer_index, series, src, note="", receiver_is_index=False):
        for k, v in series.items():
            v = float(v)
            if v == 0.0:
                continue
            entries.append((k, receiver, v, src, note) if receiver_is_index
                           else (receiver, k, v, src, note))

    # -- production
    add("commodities", "activities", sut.use, source, "intermediate use, purchaser prices")
    add("factors", "activities", sut.value_added, source, "value added")
    add_vec(gov, None, sut.production_taxes, source, "taxes on production")
    add("activities", "commodities", sut.make, source, "output, basic prices")
    add_vec(row, None, sut.imports, source, "imports")
    add_vec(gov, None, sut.product_taxes, source, "taxes on products")
    if not sut.tariffs.empty:
        add_vec(names["tariff"], None, sut.tariffs, source, "import tariffs")
        entries.append((gov, names["tariff"], float(sut.tariffs.sum()), source,
                        "tariff revenue"))
    if not sut.output_taxes.empty:
        add_vec(names["output_tax"], None, sut.output_taxes, source, "taxes on output")
        entries.append((gov, names["output_tax"], float(sut.output_taxes.sum()), source,
                        "output-tax revenue"))
    if not sut.margins.empty:
        add("commodities", "commodities", sut.margins, source, "trade and transport margins")
    if not sut.direct_sales.empty:
        add("activities", "households", sut.direct_sales, source, "own-account sales")

    # -- final demand
    for col, role in roles.items():
        target = {"households": col, "government": gov, "investment": inv,
                  "exports": row}[role]
        add_vec(target, None, sut.final[col], source, f"final demand: {col}",
                receiver_is_index=True)

    # -- institutions
    factors = list(sut.value_added.index)
    va = sut.value_added.sum(axis=1)
    if inst.factor_income is None:
        if len(hhs) != 1:
            raise ValueError(
                f"there are several household accounts {hhs}, so Institutions.factor_income "
                f"must say how each factor's income is shared among them")
        fi = pd.DataFrame({f: [float(va[f])] for f in factors}, index=hhs)
        fi_source = "assumed: all factor income to households"
    else:
        fi = _frame(inst.factor_income, "factor_income")
        fi_source = inst.source
        stray = [r for r in fi.index if r not in hhs + [gov, row]]
        if stray:
            raise KeyError(f"Institutions.factor_income pays {stray}, which are neither "
                           f"household accounts {hhs} nor {gov!r} or {row!r}")
        missing = [f for f in factors if f not in fi.columns]
        if missing:
            raise KeyError(f"Institutions.factor_income has no column for factors {missing}")
        gap = fi.sum(axis=0).reindex(factors) - va
        bad = gap[np.abs(gap) > tol * max(float(va.abs().sum()), 1.0)]
        if len(bad):
            raise ValueError("factor income paid out differs from value added for "
                             + ", ".join(f"{k} (paid − earned = {v:.6g})"
                                         for k, v in bad.items()))
    add("institutions", "factors", fi, fi_source, "factor income")
    for field_name, receiver, payer_is_hh, note in (
            ("direct_taxes", gov, True, "direct taxes"),
            ("transfers", gov, False, "government transfers"),
            ("from_abroad", row, False, "transfers from abroad"),
            ("to_abroad", row, True, "transfers abroad")):
        s = _series(getattr(inst, field_name), field_name)
        stray = [k for k in s.index if k not in hhs]
        if stray:
            raise KeyError(f"Institutions.{field_name} names {stray}, which are not "
                           f"household accounts {hhs}")
        if payer_is_hh:
            add_vec(receiver, None, s, inst.source, note)          # receiver <- household
        else:
            add_vec(receiver, None, s, inst.source, note, receiver_is_index=True)
    for flow in inst.flows:
        r, c, v = flow[0], flow[1], float(flow[2])
        entries.append((r, c, v, flow[3] if len(flow) > 3 else inst.source, "other flow"))

    # -- savings, the residuals that close each institution. Every other account
    # must already balance: a flow that unbalances one is refused by name.
    long = pd.DataFrame(entries, columns=["row", "col", "value", "source", "note"])
    receipts = long.groupby("row")["value"].sum()
    outlays = long.groupby("col")["value"].sum()
    closed_by_saving = set(hhs) | {gov, row, inv}
    gap = receipts.reindex(receipts.index.union(outlays.index), fill_value=0.0).sub(
        outlays, fill_value=0.0)
    scale = max(float(long["value"].abs().sum()), 1.0)
    bad = gap[(np.abs(gap) > tol * scale) & ~gap.index.isin(list(closed_by_saving))]
    if len(bad):
        raise ValueError(
            "accounts that no saving closes do not balance: "
            + ", ".join(f"{k} (receipts − outlays = {v:.6g})" for k, v in bad.items())
            + ". An institutional flow pays or charges one of them; record its counterpart.")
    for acct, note in [(h, "household saving") for h in hhs] + [
            (gov, "government saving"), (row, "foreign saving")]:
        saving = float(receipts.get(acct, 0.0) - outlays.get(acct, 0.0))
        if saving != 0.0:
            entries.append((inv, acct, saving, "residual", note))
    long = pd.DataFrame(entries, columns=["row", "col", "value", "source", "note"])
    funded = float(long.loc[long["row"] == inv, "value"].sum())
    invested = float(long.loc[long["col"] == inv, "value"].sum())
    scale = max(float(long["value"].abs().sum()), 1.0)
    if abs(funded - invested) > tol * scale:
        raise ValueError(
            f"saving ({funded:.6g}) and investment ({invested:.6g}) differ by "
            f"{funded - invested:.6g}: the supply and use tables and the institutional accounts "
            f"are not consistent with each other. The gap is reported, not closed.")

    # calibrate looks for the government, investment and rest-of-world accounts
    # by name; an economy without one still has it, empty.
    present = set(long["row"]) | set(long["col"])
    empty = [(a, a, 0.0, "structure", "empty account") for a in (gov, inv, row)
             if a not in present]
    if empty:
        long = pd.concat([long, pd.DataFrame(empty, columns=long.columns)], ignore_index=True)
    sam = SAM.from_long(long[["row", "col", "value"]], name=name)
    blocks = (long.groupby(["source", "note"], sort=False)
              .agg(rows=("row", lambda x: sorted(set(x))), cols=("col", lambda x: sorted(set(x))),
                   total=("value", "sum"), cells=("value", "size"))
              .reset_index())
    if sut.notes:
        extra = pd.DataFrame([dict(source=src, note=note, rows=[], cols=[], total=np.nan,
                                   cells=0) for note, src in sut.notes.items()])
        blocks = pd.concat([blocks, extra], ignore_index=True)
    sam.provenance = blocks
    return sam


# ---------------------------------------------------------------------------
# Disaggregation by household surveys
# ---------------------------------------------------------------------------

def _shares(frame, weight, group, columns, order):
    """Each group's share of the weighted total of ``columns``; None when it is zero."""
    x = frame[list(columns)].to_numpy(float).sum(axis=1) * frame[weight].to_numpy(float)
    totals = pd.Series(x).groupby(frame[group].to_numpy()).sum().reindex(order, fill_value=0.0)
    total = float(totals.sum())
    return None if abs(total) <= 1e-12 else totals / total


def group_population(survey) -> dict:
    """Persons in each group of a survey, its weights times household sizes:
    the ``population`` that ``calibrate`` takes for a split SAM."""
    f = survey.frame
    return (f[survey.weight] * f[survey.size]).groupby(f[survey.group]).sum().to_dict()


def split_households(sam: SAM, survey, *, account: str, incomes: Mapping | None = None,
                     outlays: Mapping | None = None, fallback: Mapping | None = None,
                     names: Mapping[str, str] | None = None,
                     source: str = "household survey") -> SAM:
    """Split one household account into the groups of a survey.

    Every cell of the account is shared among the survey's groups (its
    ``group`` column) by each group's share of the weighted survey total for
    that cell: the income from a factor by the columns the survey maps to it
    (``Survey.sources``; ``"transfers"`` stands for government and
    ``"remittances"`` for the rest of the world), the spending on a commodity by
    its budget columns. ``incomes`` and ``outlays`` name the columns for any
    other payer or receiver, a direct tax, say: ``{account: column or list}``.
    ``fallback`` says how to share a cell the survey has nothing on:
    ``"population"`` (weights times sizes) or ``"households"`` (weights). A cell
    with neither is refused, by name.

    Each group's saving is what is left, and a negative one is data, kept. The
    SAM's totals are unchanged: summing the groups gives back the account.
    """
    names = {**_NAMES, **(names or {})}
    gov, inv, row = names["government"], names["investment"], names["rest_of_world"]
    M = sam.matrix()
    if account not in M.index:
        raise KeyError(f"{account!r} is not an account of the SAM")
    f = survey.frame
    order = list(pd.unique(f[survey.group]))
    clash = [g for g in order if g in M.index and g != account]
    if clash:
        raise ValueError(f"survey groups {clash} are already accounts of the SAM")
    if M.at[account, account] != 0:
        raise ValueError(f"{account!r} pays itself; that cell has no survey counterpart")
    special = {"transfers": gov, "remittances": row}

    def payer_of(src):
        return special.get(src, src.split(":")[0])

    def columns_for(cell, is_income):
        given = (incomes if is_income else outlays) or {}
        if cell in given:
            cols = given[cell]
            return [cols] if isinstance(cols, str) else list(cols)
        if is_income:
            return [c for c, s in survey.sources.items() if payer_of(s) == cell]
        return [c for c, com in (survey.budget or {}).items() if com == cell]

    missing, plan = [], []
    for is_income, cells in ((True, [p for p in M.columns if M.at[account, p] != 0]),
                             (False, [r for r in M.index if M.at[r, account] != 0 and r != inv])):
        for cell in cells:
            cols = columns_for(cell, is_income)
            shares = _shares(f, survey.weight, survey.group, cols, order) if cols else None
            how = f"{'income from' if is_income else 'spending on'} {cell}: {', '.join(cols)}"
            if shares is None:
                rule = (fallback or {}).get(cell)
                if rule in ("population", "households"):
                    unit = (f[survey.size].astype(float) if rule == "population"
                            else pd.Series(1.0, index=f.index))
                    shares = _shares(f.assign(_unit=unit), survey.weight, survey.group,
                                     ["_unit"], order)
                    how = f"{'income from' if is_income else 'spending on'} {cell}: by {rule}"
                else:
                    missing.append(cell)
                    continue
            plan.append((is_income, cell, shares, how))
    if missing:
        raise ValueError(
            f"the survey has nothing to share {account!r}'s cells with {missing}: map a column "
            f"to them (Survey.sources or budget, or incomes= / outlays=), or give a fallback")

    N = M.drop(index=account, columns=account)
    for g in order:
        N.loc[g] = 0.0
        N[g] = 0.0
    prov = []
    for is_income, cell, shares, how in plan:
        v = float(M.at[account, cell] if is_income else M.at[cell, account])
        for g in order:
            if is_income:
                N.at[g, cell] = v * float(shares[g])
            else:
                N.at[cell, g] = v * float(shares[g])
        prov.append((source, how, order if is_income else [cell],
                     [cell] if is_income else order, v, len(order)))
    for g in order:
        saving = float(N.loc[g].sum() - N[g].drop(index=inv, errors="ignore").sum())
        N.at[inv, g] = saving
        prov.append(("residual", f"saving of {g}" + (" (negative)" if saving < 0 else ""),
                     [inv], [g], saving, 1))
    out = _square(N.fillna(0.0), sam.name)
    out.provenance = pd.DataFrame(prov, columns=["source", "note", "rows", "cols", "total",
                                                 "cells"])
    return out


def segment_shares(survey, persons, *, segments: Sequence[str],
                   activity: str = "activity") -> tuple:
    """Earnings of each labour segment by activity, and by survey group.

    From a household survey and its persons (``cgelab.ibge.pnadc``): the
    weighted earnings of the persons in each segment, by the SAM activity of
    their main job and by the group of their household. They are what
    :func:`split_factor` takes, ``(by_activity, owners)``.
    """
    pf = persons.frame
    f = survey.frame
    if activity not in pf.columns:
        raise KeyError(f"the persons have no {activity!r} column: read the PNAD Contínua with "
                       f"activities=")
    at = f.index.get_indexer(pf[persons.household])
    if np.any(at < 0):
        raise ValueError("persons belong to households not in the survey")
    w = f[survey.weight].to_numpy(float)[at]
    groups = f[survey.group].to_numpy()[at]
    status = pf[persons.status].astype(str).to_numpy()
    y = pf[persons.earnings].to_numpy(float) * w
    work = pd.DataFrame({"segment": status, "activity": pf[activity].to_numpy(),
                         "group": groups, "y": y})
    work = work[work["segment"].isin(list(segments))]
    by_activity = (work.pivot_table(index="segment", columns="activity", values="y",
                                    aggfunc="sum", fill_value=0.0)
                   .reindex(list(segments), fill_value=0.0))
    owners = (work.pivot_table(index="group", columns="segment", values="y", aggfunc="sum",
                               fill_value=0.0)
              .reindex(columns=list(segments), fill_value=0.0))
    return by_activity, owners


def split_factor(sam: SAM, factor: str, by_activity: pd.DataFrame, *,
                 owners: pd.DataFrame | None = None, names: Mapping[str, str] | None = None,
                 source: str = "household survey") -> SAM:
    """Split a factor into several (formal and informal labour, say).

    ``by_activity`` is new factors by activities: in each activity the factor's
    payment is shared in proportion to its column. The new factors' income goes
    to the factor's recipients: government and the rest of the world keep their
    shares of each, and the households' part follows ``owners``, recipients by
    new factors, in proportion to each column; without ``owners`` every
    recipient keeps its share of each new factor. When ``owners`` moves income
    between households, their savings are recomputed, as residuals.

    Summing the new factors gives back the factor's row and column.
    """
    names = {**_NAMES, **(names or {})}
    inv = names["investment"]
    M = sam.matrix()
    if factor not in M.index:
        raise KeyError(f"{factor!r} is not an account of the SAM")
    if not isinstance(by_activity, pd.DataFrame):
        raise TypeError("by_activity must be a DataFrame, new factors by activities")
    new = list(by_activity.index)
    clash = [k for k in new if k in M.index]
    if clash:
        raise ValueError(f"new factors {clash} are already accounts of the SAM")
    payers = [a for a in M.columns if M.at[factor, a] != 0]
    lacking = [a for a in payers if a not in by_activity.columns
               or float(by_activity[a].sum()) <= 0]
    if lacking:
        raise ValueError(f"by_activity has no positive shares for {lacking}, which pay "
                         f"{factor!r}")
    shares = by_activity[payers].astype(float)
    shares = shares / shares.sum(axis=0)
    recipients = [r for r in M.index if M.at[r, factor] != 0]
    income = float(M[factor].sum())
    s = {r: float(M.at[r, factor]) / income for r in recipients}

    N = M.drop(index=factor, columns=factor)
    for k in new:
        N.loc[k] = 0.0
        N[k] = 0.0
    prov = []
    for k in new:
        for a in payers:
            N.at[k, a] = float(M.at[factor, a]) * float(shares.at[k, a])
    prov.append((source, f"{factor} split by activity", new, payers,
                 float(M.loc[factor, payers].sum()), len(new) * len(payers)))
    inc = {k: float(N.loc[k, payers].sum()) for k in new}
    touched = []
    if owners is None:
        for k in new:
            for r in recipients:
                N.at[r, k] = s[r] * inc[k]
        prov.append(("SAM", f"{factor} income, shares of the original recipients", recipients,
                     new, income, len(recipients) * len(new)))
    else:
        owners = owners.astype(float).reindex(columns=new, fill_value=0.0)
        stray = [r for r in owners.index if r not in recipients]
        if stray:
            raise ValueError(f"owners {stray} receive no {factor!r} income in the SAM")
        empty = [k for k in new if float(owners[k].sum()) <= 0]
        if empty:
            raise ValueError(f"owners has no positive entries for {empty}")
        others = [r for r in recipients if r not in owners.index]
        pool = 1.0 - sum(s[r] for r in others)
        cols = owners / owners.sum(axis=0)
        for k in new:
            for r in others:
                N.at[r, k] = s[r] * inc[k]
            for r in owners.index:
                N.at[r, k] = pool * inc[k] * float(cols.at[r, k])
        prov.append((source, f"{factor} income of households, by owners", list(owners.index),
                     new, pool * income, len(owners.index) * len(new)))
        touched = list(owners.index)
    for h in touched:
        N.at[inv, h] = 0.0
        saving = float(N.loc[h].sum() - N[h].sum())
        N.at[inv, h] = saving
        prov.append(("residual", f"saving of {h}, recomputed", [inv], [h], saving, 1))
    out = _square(N.fillna(0.0), sam.name)
    out.provenance = pd.DataFrame(prov, columns=["source", "note", "rows", "cols", "total",
                                                 "cells"])
    return out


def _square(N: pd.DataFrame, name: str) -> SAM:
    """A SAM from a square frame that keeps every account, an all-zero one too.

    A government or a rest of the world with no flows is still an account the
    model's layout names.
    """
    long = N.stack().reset_index()
    long.columns = ["row", "col", "value"]
    nonzero = long[long.value != 0.0]
    seen = set(nonzero.row) | set(nonzero.col)
    empty = [a for a in N.index if a not in seen]
    keep = pd.DataFrame({"row": empty, "col": empty, "value": [0.0] * len(empty)})
    return SAM.from_long(pd.concat([nonzero, keep], ignore_index=True), name=name)


def split_vintages(sam: SAM, share, *, capital: str = "f_cap", new: str = "f_cap_new",
                   source: str = "assumed") -> SAM:
    """Split a capital account into installed capital and capital bought this year.

    ``share`` is the new vintage's part of each activity's capital payment: one
    number, or a mapping by activity (``"__default__"`` for the rest). The
    capital account keeps the old vintage's part and ``new`` gets the rest;
    every recipient of capital income keeps its share of both. For
    ``ModelSpec(vintages={capital: new})``.

    In ``run_path`` the new vintage is a year's investment, so a base year on its
    steady path has, as a stock, new capital equal to investment: ``share`` is
    the investment over the capital stock, ``(depreciation + growth)`` on a
    balanced path, or ``user_cost * investment / capital payments`` read off the
    SAM.
    """
    M = sam.matrix()
    if capital not in M.index:
        raise KeyError(f"{capital!r} is not an account of the SAM")
    if new in M.index:
        raise ValueError(f"{new!r} is already an account of the SAM")
    payers = [a for a in M.columns if M.at[capital, a] != 0]
    if isinstance(share, Mapping):
        default = share.get("__default__")
        missing = [a for a in payers if a not in share and default is None]
        if missing:
            raise ValueError(f"share has no value for {missing}, which pay {capital!r}; "
                             f"give one or a \"__default__\"")
        values = [float(share.get(a, default)) for a in payers]
    else:
        values = [float(share)] * len(payers)
    bad = [a for a, v in zip(payers, values) if not 0.0 <= v <= 1.0]
    if bad:
        raise ValueError(f"the new vintage's share must be between 0 and 1, not in {bad}")
    old = f"{capital} (old)"
    by = pd.DataFrame([[1.0 - v for v in values], values], index=[old, new], columns=payers)
    # split_factor needs a share for each vintage in each activity; an activity
    # with all of its capital in one vintage gets a zero on the other.
    out = split_factor(sam, capital, by, source=source)
    rename = {old: capital}
    out_m = out.matrix().rename(index=rename, columns=rename)
    prov = out.provenance.copy()
    prov["note"] = prov["note"].str.replace(old, capital, regex=False)
    for col in ("rows", "cols"):
        prov[col] = prov[col].apply(lambda xs: [rename.get(x, x) for x in xs])
    prov.loc[0, "note"] = f"{capital} split into vintages: {new} the new one"
    result = _square(out_m, sam.name)
    result.provenance = prov
    return result


# ---------------------------------------------------------------------------
# The IBGE's supply and use tables
# ---------------------------------------------------------------------------

def _excel(path, sheet):
    """One sheet of an IBGE workbook, or of a ``{sheet: DataFrame}`` already read."""
    if isinstance(path, Mapping):
        if sheet not in path:
            raise KeyError(f"no sheet {sheet!r} among {sorted(path)}")
        return path[sheet].astype(str).replace({"nan": ""})
    try:
        return pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    except ImportError as err:                         # pragma: no cover - depends on the install
        raise ImportError(
            f"reading {path} needs an Excel engine for pandas: pip install xlrd (for .xls) "
            f"or openpyxl (.xlsx), or odfpy (.ods)") from err


def _numbers(block) -> pd.DataFrame:
    return block.apply(lambda c: pd.to_numeric(c.astype(str).str.strip().replace({"": "0",
                                                                               "nan": "0"}),
                                               errors="coerce")).fillna(0.0).astype(float)


def _ibge_block(sheet, first_col=2, last_col=None):
    """The product rows of an IBGE table: codes in column 0, values from ``first_col``."""
    codes = sheet.iloc[:, 0].fillna("").astype(str).str.strip()
    rows = [i for i, c in enumerate(codes) if c.isdigit()]
    if not rows:
        raise ValueError("no product rows (numeric codes in the first column) in the sheet")
    end = sheet.shape[1] if last_col is None else last_col
    values = _numbers(sheet.iloc[rows, first_col:end])
    values.index = codes.iloc[rows].tolist()
    return values, sheet.iloc[rows, 1].fillna("").astype(str).str.strip().tolist()


def _activity_header(sheet, n):
    """Activity codes and names from the header cells ``'0191\\nAgricultura...'``."""
    out = []
    for cell in sheet.iloc[3, 2:2 + n].tolist():
        text = " ".join(str(cell).split())
        code, _, label = text.partition(" ")
        if not code.isdigit():
            raise ValueError(f"an activity header does not start with its code: {text!r}")
        out.append((code, label))
    return out


def read_tru(supply_path, use_path, *, activity_prefix: str = "a_", commodity_prefix: str = "c_",
             households: str = "hh", npish: str = "households", mixed_income: str = "separate",
             tariffs: bool = False, factors: Mapping[str, str] | None = None) -> SupplyUse:
    """The IBGE's supply and use tables (Sistema de Contas Nacionais) as a ``SupplyUse``.

    ``supply_path`` is table 1 (*Recursos de bens e serviços*: sheets
    ``oferta``, ``producao``, ``importacao``) and ``use_path`` table 2 (*Usos
    de bens e serviços*: ``CI``, ``demanda``, ``VA``), at current prices, of any
    level (12, 20, 51 or 68 activities); values are in R$ million. Either may be
    a ``{sheet: DataFrame}`` already read, with no header row, as
    ``pd.read_excel(..., header=None)`` gives it; reading the ``.xls`` files
    needs ``xlrd``. Accounts are the IBGE's codes with ``activity_prefix`` and
    ``commodity_prefix``.

    * Labour is *Remunerações*; capital is *Excedente operacional bruto*. Mixed
      income (*Rendimento misto bruto*) is a factor of its own with
      ``mixed_income="separate"`` (``factors["mixed"]``, ``"f_mix"``), or joins
      ``"labour"`` or ``"capital"``: a choice with distributional weight, made
      explicitly.
    * Taxes on products are II, IPI, ICMS and other taxes net of subsidies; with
      ``tariffs=True`` the import tax goes to a tariff account. Other taxes on
      production, net of subsidies, are taxes on production.
    * The trade and transport margin columns are positive on the goods that bear
      them and negative on the products that provide them; each good's margin is
      shared among the providers in proportion to what each provides, a
      proportional allocation the provenance records.
    * Final demand: households, government, investment (*FBCF* and *variação de
      estoque*) and exports; the NPISH (*ISFLSF*) join ``npish``,
      ``"households"`` or ``"government"``.

    Supply at purchaser prices must equal total demand for every product, which
    ``sut_to_sam`` checks; the published tables are integers, so pass it a
    tolerance that covers their rounding.
    """
    if mixed_income not in ("separate", "labour", "capital"):
        raise ValueError("mixed_income must be 'separate', 'labour' or 'capital'")
    if npish not in ("households", "government"):
        raise ValueError("npish must be 'households' or 'government'")
    fac = {"labour": "f_lab", "capital": "f_cap", "mixed": "f_mix", **(factors or {})}

    oferta = _excel(supply_path, "oferta")
    producao = _excel(supply_path, "producao")
    importacao = _excel(supply_path, "importacao")
    ci = _excel(use_path, "CI")
    demanda = _excel(use_path, "demanda")
    va = _excel(use_path, "VA")

    offer, names = _ibge_block(oferta)
    coms = [commodity_prefix + c for c in offer.index]
    labels = dict(zip(coms, names))
    head = [" ".join(str(c).split()) for c in oferta.iloc[3].tolist()]

    def offer_col(title):
        hits = [i for i, h in enumerate(head) if h.lower().startswith(title.lower())]
        if len(hits) != 1:
            raise ValueError(f"the 'oferta' sheet has no single column {title!r}: {head}")
        return offer.iloc[:, hits[0] - 2].to_numpy()

    n_act = sum(1 for c in producao.iloc[3, 2:].tolist()
                if " ".join(str(c).split())[:4].isdigit())
    acts_raw = _activity_header(producao, n_act)
    acts = [activity_prefix + c for c, _ in acts_raw]
    labels.update({activity_prefix + c: label for c, label in acts_raw})
    if [c for c, _ in _activity_header(ci, n_act)] != [c for c, _ in acts_raw]:
        raise ValueError("the activities of 'CI' and 'producao' differ")

    prod, _ = _ibge_block(producao, 2, 2 + n_act)
    make = pd.DataFrame(prod.to_numpy().T, index=acts, columns=coms)
    inter, _ = _ibge_block(ci, 2, 2 + n_act)
    use = pd.DataFrame(inter.to_numpy(), index=coms, columns=acts)
    imp, _ = _ibge_block(importacao, 2, 3)
    imports = pd.Series(imp.iloc[:, 0].to_numpy(), index=coms)

    dem, _ = _ibge_block(demanda, 2)
    dhead = [" ".join(str(c).split()).lower() for c in demanda.iloc[3].tolist()]

    def dem_col(title):
        hits = [i for i, h in enumerate(dhead) if h.startswith(title)]
        if len(hits) != 1:
            raise ValueError(f"the 'demanda' sheet has no single column {title!r}")
        return dem.iloc[:, hits[0] - 2].to_numpy()

    final = pd.DataFrame({households: dem_col("consumo das famílias"),
                          "gov": dem_col("consumo do governo"),
                          "s_i": dem_col("formação bruta de capital fixo")
                          + dem_col("variação de estoque"),
                          "row": dem_col("exportação")}, index=coms)
    final[households if npish == "households" else "gov"] += dem_col("consumo das isflsf")

    labels_va = [" ".join(str(c).split()).lower() for c in va.iloc[:, 0].tolist()]

    def va_row(title):
        hits = [i for i, h in enumerate(labels_va) if h.startswith(title)]
        if len(hits) != 1:
            raise ValueError(f"the 'VA' sheet has no single row {title!r}")
        return _numbers(va.iloc[[hits[0]], 1:1 + n_act]).to_numpy().ravel()

    if [" ".join(str(c).split())[:4] for c in va.iloc[3, 1:1 + n_act].tolist()] != \
            [c for c, _ in acts_raw]:
        raise ValueError("the activities of 'VA' and 'producao' differ")
    labour = va_row("remunerações")
    surplus = va_row("excedente operacional bruto (eob)")
    mixed = va_row("rendimento misto bruto")
    rows = {fac["labour"]: labour, fac["capital"]: surplus}
    if mixed_income == "separate":
        rows[fac["mixed"]] = mixed
    else:
        rows[fac[mixed_income]] = rows[fac[mixed_income]] + mixed
    value_added = pd.DataFrame(rows, index=acts).T
    production_taxes = pd.Series(va_row("outros impostos sobre a produção")
                                 + va_row("outros subsídios à produção"), index=acts)

    ii = offer_col("imposto de importação")
    taxes = offer_col("ipi") + offer_col("icms") + offer_col("outros impostos")
    product_taxes = pd.Series(taxes + (0.0 if tariffs else ii), index=coms)
    tariff = pd.Series(ii, index=coms) if tariffs else None

    margins = pd.DataFrame(0.0, index=coms, columns=coms)
    for title in ("margem de comércio", "margem de transporte"):
        m = offer_col(title)
        providers = np.where(m < 0)[0]
        if not len(providers):
            continue
        provided = m[providers]
        share = provided / provided.sum()
        bearers = np.where(m > 0)[0]
        for k, j in enumerate(providers):
            margins.iloc[j, bearers] += m[bearers] * share[k]
            margins.iloc[j, j] += m[j]
    margins = margins.loc[(margins != 0).any(axis=1)]

    return SupplyUse(
        make=make, use=use, final=final, imports=imports, value_added=value_added,
        product_taxes=product_taxes, production_taxes=production_taxes, margins=margins,
        tariffs=tariff, labels=labels,
        notes={"margins shared among their providers in proportion to what each provides":
               "assumed by read_tru",
               f"mixed income: {mixed_income}": "read_tru",
               f"NPISH consumption joins {npish}": "read_tru"})
