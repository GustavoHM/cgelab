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

__all__ = ["SupplyUse", "Institutions", "sut_to_sam"]

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
    sam.provenance = blocks
    return sam
