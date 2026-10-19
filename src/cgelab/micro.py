"""From the household accounts of the model to the households of a survey.

The SAM has a few household accounts; poverty and inequality live in the
thousands of households a survey records. This module carries a solved
counterfactual to them top-down, without behaviour of their own: each survey
household belongs to one account, its income from each source moves as that
account's income from the source moves in the model, and its cost of living
moves with its own budget at the prices its account pays.

Three choices are explicit, because each changes a distributional number.

* **Reconciliation.** A survey never adds up to the SAM. By default each
  source of each account is scaled so that the weighted survey total is the
  SAM's, and the factors are reported in ``MicroResult.reconciliation``: the
  discrepancy is information about the data and is not folded away silently.
  Without it the survey keeps its own levels and only the relative changes
  are applied, so the micro totals no longer add up to the model.
* **The price index** is geometric in the household's own budget shares, the
  exact index for Cobb–Douglas and the one the model uses for its accounts, so
  a survey that is the SAM gives back the model's real incomes. Laspeyres is
  available; it is never below the geometric index.
* **The welfare measure** is real disposable income per person, or per adult
  equivalent with ``size ** θ``. The poverty line is in the reference period's
  prices.

The measures (Gini, FGT, generalised entropy and Atkinson) take any vector and
weights, and are used on their own as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["gini", "fgt", "generalized_entropy", "theil", "mean_log_deviation", "atkinson",
           "ge_decomposition", "fgt_decomposition", "Survey", "MicroResult", "microsimulate",
           "account_sources", "inequality", "poverty"]


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def _prep(x, weights, *, positive=False, what="measure"):
    x = np.asarray(x, dtype=float).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != x.shape:
        raise ValueError(f"weights of shape {w.shape} do not match incomes {x.shape}")
    if x.size == 0:
        raise ValueError("no observations")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w))):
        raise ValueError("incomes and weights must be finite")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    if positive and np.any(x[w > 0] <= 0):
        n = int(np.sum(x[w > 0] <= 0))
        raise ValueError(
            f"the {what} needs positive incomes; {n} observation(s) are zero or negative. "
            f"Drop them, or use the Gini or FGT, which accept them.")
    return x, w


def gini(x, weights=None) -> float:
    """Gini coefficient, from the Lorenz curve of the weighted distribution.

    With integer weights it is the Gini of the sample with each observation
    repeated that many times; it equals the mean absolute difference over twice
    the mean. Incomes may be zero or negative, but their mean must be positive.
    """
    x, w = _prep(x, weights)
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    total = float(w @ x)
    if total <= 0:
        raise ValueError("the Gini needs a positive mean income")
    p = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    L = np.concatenate([[0.0], np.cumsum(w * x) / total])
    return float(1.0 - np.sum(np.diff(p) * (L[1:] + L[:-1])))


def fgt(x, line, alpha=0, weights=None) -> float:
    """Foster–Greer–Thorbecke poverty index: the weighted mean of ``(1 − x/z)^α`` below ``z``.

    ``alpha = 0`` is the headcount ratio, 1 the poverty gap, 2 the squared gap.
    Poor means strictly below the line.
    """
    x, w = _prep(x, weights)
    z = float(line)
    if not z > 0:
        raise ValueError(f"the poverty line must be positive, got {line}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    poor = x < z
    gap = np.where(poor, 1.0 - x / z, 0.0)
    terms = np.where(poor, 1.0, 0.0) if alpha == 0 else np.where(poor, gap ** alpha, 0.0)
    return float(w @ terms / w.sum())


def generalized_entropy(x, alpha=1.0, weights=None) -> float:
    """Generalised entropy index GE(α).

    GE(0) is the mean log deviation (Theil's L), GE(1) Theil's T, and GE(2)
    half the squared coefficient of variation. Incomes must be positive.
    """
    x, w = _prep(x, weights, positive=True, what="generalised entropy")
    p = w / w.sum()
    r = x / float(p @ x)
    a = float(alpha)
    if a == 0.0:
        return float(-(p @ np.log(r)))
    if a == 1.0:
        return float(p @ (r * np.log(r)))
    return float((p @ r ** a - 1.0) / (a * (a - 1.0)))


def theil(x, weights=None) -> float:
    """Theil's T index, GE(1)."""
    return generalized_entropy(x, 1.0, weights)


def mean_log_deviation(x, weights=None) -> float:
    """Mean log deviation, Theil's L, GE(0)."""
    return generalized_entropy(x, 0.0, weights)


def atkinson(x, epsilon=1.0, weights=None) -> float:
    """Atkinson index A(ε): one minus the equally distributed equivalent income over the mean.

    ε = 0 is indifference to inequality (zero); ε = 1 uses the geometric mean.
    Incomes must be positive for ε > 0.
    """
    e = float(epsilon)
    if e < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    x, w = _prep(x, weights, positive=e > 0, what="Atkinson index")
    p = w / w.sum()
    mu = float(p @ x)
    if e == 1.0:
        ede = float(np.exp(p @ np.log(x)))
    else:
        ede = float((p @ x ** (1.0 - e)) ** (1.0 / (1.0 - e)))
    return 1.0 - ede / mu


def ge_decomposition(x, alpha, groups, weights=None) -> pd.DataFrame:
    """GE(α) split into the inequality within groups and between them.

    Within is ``Σ_g v_g^(1−α) s_g^α GE_g``, with ``v_g`` the group's population
    share and ``s_g`` its income share; between is GE(α) of everyone given their
    group's mean. Rows are the groups, then ``within``, ``between`` and ``total``.
    """
    x, w = _prep(x, weights, positive=True, what="generalised entropy")
    g = np.asarray(groups).ravel()
    if g.shape != x.shape:
        raise ValueError("groups must have one entry per observation")
    a = float(alpha)
    W, Y = w.sum(), float(w @ x)
    rows, means = {}, np.empty_like(x)
    for name in pd.unique(g):
        m = g == name
        v, s = w[m].sum() / W, float(w[m] @ x[m]) / Y
        ge_g = generalized_entropy(x[m], a, w[m])
        rows[name] = dict(population_share=v, income_share=s, index=ge_g,
                          contribution=v ** (1.0 - a) * s ** a * ge_g)
        means[m] = float(w[m] @ x[m]) / w[m].sum()
    out = pd.DataFrame.from_dict(rows, orient="index")
    within = float(out["contribution"].sum())
    between = generalized_entropy(means, a, w)
    total = generalized_entropy(x, a, w)
    extra = pd.DataFrame({"contribution": [within, between, total]},
                         index=["within", "between", "total"])
    return pd.concat([out, extra])


def fgt_decomposition(x, line, alpha, groups, weights=None) -> pd.DataFrame:
    """FGT(α) by group: each group's index, population share and contribution.

    The FGT family is additively decomposable: the total is the
    population-weighted sum of the groups' indices, which the ``total`` row shows.
    """
    x, w = _prep(x, weights)
    g = np.asarray(groups).ravel()
    if g.shape != x.shape:
        raise ValueError("groups must have one entry per observation")
    W = w.sum()
    rows = {}
    for name in pd.unique(g):
        m = g == name
        v = w[m].sum() / W
        f = fgt(x[m], line, alpha, w[m])
        rows[name] = dict(population_share=v, index=f, contribution=v * f)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.loc["total"] = [1.0, fgt(x, line, alpha, w), float(out["contribution"].sum())]
    return out


# ---------------------------------------------------------------------------
# What each household account earns, by source, in a solved period
# ---------------------------------------------------------------------------

def account_sources(period) -> pd.DataFrame:
    """Each household account's gross income by source, in a solved period.

    Columns: every factor (its income net of what government and the rest of
    the world receive), ``"factor:activity"`` for each activity that uses the
    factor, ``"transfers"`` (from government and the scenario, less the levy
    that finances fixed real investment), ``"remittances"`` (net, from abroad)
    and ``"recycled"`` (wedge revenue returned to households). The factor
    columns and the last three add up to ``period.hh_income``; the
    ``"factor:activity"`` columns split each factor column by activity.
    """
    from .mobility import period_endowments
    from .solve import _recycle_shares

    econ, sc, closure = period.econ, period.scenario, period.closure
    if econ.regional is not None:
        raise ValueError("microsimulation is not available for a regional model yet")
    nu = closure.numeraire
    nh, nf = econ.n_hh, econ.n_fac
    facshare = (econ.hh_facshare if sc.factor_ownership is None
                else np.asarray(sc.factor_ownership, float))
    gov_sh = econ.zeros_if_none("fac_gov_share", nf)
    row_sh = (econ.zeros_if_none("fac_row_share", nf) if sc.foreign_factor_share is None
              else np.asarray(sc.foreign_factor_share, float))
    keep = 1.0 - gov_sh - row_sh
    by_factor = np.zeros((nh, nf))
    grouped = set()
    for mk in (period.labour_markets or {}).values():
        for k, f in enumerate(mk["segments"]):
            i = econ.factor(f)
            grouped.add(i)
            by_factor[:, i] = mk["household_employment"][:, k] * mk["wage"][k] * keep[i]
    # A factor is paid for what it supplies. Every market but the numeraire's
    # clears, so there that is the demand; the numeraire's market is the Walras
    # check, and it is paid its supply, as the solver pays it.
    supply, _ = period_endowments(econ, sc)
    income = period.factor_prices * period.factor_demand
    num = econ.numeraire_index
    income[num] = period.factor_prices[num] * supply[num]
    for i in range(nf):
        if i not in grouped:
            by_factor[:, i] = facshare[:, i] * income[i] * keep[i]
    out = pd.DataFrame(by_factor, index=list(econ.hhs), columns=list(econ.factors))

    # Each factor's income split by the activity that pays it.
    w_act = (period.factor_prices_by_activity if period.factor_prices_by_activity is not None
             else np.tile(period.factor_prices[:, None], (1, econ.n_act)))
    paid = w_act * period.factor_demand_by_activity
    used = (econ.aunit * econ.X0[None, :]) > 0
    for i, f in enumerate(econ.factors):
        tot = float(paid[i].sum())
        for j, a in enumerate(econ.acts):
            if used[i, j]:
                out[f"{f}:{a}"] = by_factor[:, i] * (paid[i, j] / tot if tot else 0.0)

    base_tr = np.zeros(nh) if sc.transfers is None else np.asarray(sc.transfers, float)
    out["transfers"] = (base_tr + econ.zeros_if_none("hh_gov_transfer", nh) * nu
                        - period.forced_saving * econ.pop_share)
    out["remittances"] = econ.zeros_if_none("hh_remit_net", nh) * nu
    out["recycled"] = period.wedge_revenue * _recycle_shares(econ, sc, facshare, supply)
    return out


def _total_columns(econ) -> list:
    return list(econ.factors) + ["transfers", "remittances", "recycled"]


# ---------------------------------------------------------------------------
# The survey
# ---------------------------------------------------------------------------

@dataclass
class Survey:
    """A household survey, one row per household, mapped onto the model.

    ``sources`` maps an income column to what moves it: a factor (``"f_cap"``),
    a factor in one activity (``"f_lab:a_agr"``), ``"transfers"``,
    ``"remittances"``, ``"recycled"``, or ``"fixed"`` (held in nominal terms and
    not reconciled). Several columns may share a source. ``budget`` maps a
    spending column to a commodity; absent, each household uses its account's
    marginal budget shares. ``workers`` maps a head-count column to a labour
    segment or ``"unemployed"``; it is read by the reweighting, not here.
    """

    frame: pd.DataFrame
    sources: Mapping[str, str]
    group: str = "group"
    weight: str = "weight"
    size: str = "size"
    budget: Mapping[str, str] | None = None
    workers: Mapping[str, str] | None = None

    def __post_init__(self):
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("Survey.frame must be a DataFrame, one row per household")
        if not self.sources:
            raise ValueError("Survey.sources is empty: say which columns are income and from "
                             "what source")
        cols = [self.group, self.weight, self.size, *self.sources, *(self.budget or {}),
                *(self.workers or {})]
        missing = [c for c in cols if c not in self.frame.columns]
        if missing:
            raise KeyError(f"columns not in the survey: {missing}")
        w = self.frame[self.weight].to_numpy(float)
        n = self.frame[self.size].to_numpy(float)
        if not (np.all(np.isfinite(w)) and np.all(w > 0)):
            raise ValueError(f"survey weights ({self.weight!r}) must be positive and finite")
        if not (np.all(np.isfinite(n)) and np.all(n > 0)):
            raise ValueError(f"household sizes ({self.size!r}) must be positive and finite")
        if self.budget is not None and np.any(self.frame[list(self.budget)].to_numpy(float) < 0):
            raise ValueError("budget columns must be non-negative spending")

    def check(self, econ) -> None:
        """Refuse what does not match the economy, by name."""
        stray = sorted(set(self.frame[self.group]) - set(econ.hhs))
        if stray:
            raise ValueError(f"survey groups {stray} are not household accounts of the SAM "
                             f"({list(econ.hhs)})")
        known = set(econ.factors) | {"transfers", "remittances", "recycled", "fixed"}
        used = (econ.aunit * econ.X0[None, :]) > 0
        known |= {f"{f}:{a}" for i, f in enumerate(econ.factors)
                  for j, a in enumerate(econ.acts) if used[i, j]}
        bad = {c: s for c, s in self.sources.items() if s not in known}
        if bad:
            raise ValueError(f"unknown income sources {bad}: a source is a factor, "
                             f"'factor:activity' for an activity that uses it, 'transfers', "
                             f"'remittances', 'recycled' or 'fixed'")
        whole = {s for s in self.sources.values() if ":" not in s}
        parts = {s.split(":")[0] for s in self.sources.values() if ":" in s}
        both = sorted(whole & parts)
        if both:
            raise ValueError(f"factors {both} are mapped both whole and by activity; the "
                             f"same income would be scaled twice")
        if self.budget is not None:
            bad = {c: s for c, s in self.budget.items() if s not in econ.coms}
            if bad:
                raise ValueError(f"budget columns {bad} do not name commodities of the SAM")


# ---------------------------------------------------------------------------
# Microsimulation
# ---------------------------------------------------------------------------

@dataclass
class MicroResult:
    """A counterfactual carried to the households of a survey."""

    #: One row per household: ``group``, ``weight``, ``size``, gross and
    #: disposable income in the reference and the alternative, ``price_index``
    #: (the alternative's cost of living over the reference's) and real
    #: disposable income, ``real_ref`` and ``real_alt``, at reference prices.
    frame: pd.DataFrame
    #: Income by household and source, reconciled, in the two periods.
    income_ref: pd.DataFrame = field(repr=False)
    income_alt: pd.DataFrame = field(repr=False)
    #: ``(group, source)``: the survey's weighted total, the SAM account's, and
    #: the factor that scaled the survey to it. Empty without reconciliation.
    reconciliation: pd.DataFrame = field(repr=False)
    #: ``(group, constraint)``: the reweighting's base total, target and the
    #: total the new weights reach, and its iterations. Empty when nothing was
    #: reweighted.
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    #: ``(group, column)``: persons in each ``Survey.workers`` column, with the
    #: survey weights and with the new ones.
    workers: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def welfare(self, equivalence: float = 1.0) -> pd.DataFrame:
        """Real disposable income per adult equivalent, ``size ** equivalence``,
        and person weights in each period: the inputs of every measure."""
        if not 0.0 <= float(equivalence) <= 1.0:
            raise ValueError(f"equivalence must lie in [0, 1], got {equivalence}")
        f = self.frame
        scale = f["size"] ** float(equivalence)
        return pd.DataFrame({"group": f["group"],
                             "persons_ref": f["weight"] * f["size"],
                             "persons_alt": f["weight_alt"] * f["size"],
                             "ref": f["real_ref"] / scale, "alt": f["real_alt"] / scale})


def microsimulate(survey: Survey, reference, alternative, *, reconcile: bool = True,
                  reweight: bool = True, price_index: str = "geometric") -> MicroResult:
    """Carry the change from ``reference`` to ``alternative`` to the survey's households.

    Both are solved periods of the same economy. The survey is taken to
    describe the reference. Each household's income from a source moves as its
    account's income from that source does; a source its account did not have
    in the reference and has in the alternative is shared per person among the
    account's households. Disposable income takes the account's direct tax rate.

    With ``reweight`` and labour segments (``ModelSpec.labour_segments``), who
    works where changes through the survey weights instead: within each account,
    the weights are calibrated (raking, Deville and Särndal 1992) so that the
    survey's employment in each segment moves as the account's does in the
    model, its unemployed persons as the account's queue does, and every other
    source, and the number of households, stays where it was. Each segment's
    income per worker then moves with its wage, and the micro totals are still
    the model's.
    """
    econ = reference.econ
    if alternative.econ.hhs != econ.hhs or alternative.econ.coms != econ.coms:
        raise ValueError("reference and alternative must come from the same economy")
    if price_index not in ("geometric", "laspeyres"):
        raise ValueError(f"price_index must be 'geometric' or 'laspeyres', got {price_index!r}")
    survey.check(econ)
    f = survey.frame
    grp = f[survey.group].to_numpy()
    w = f[survey.weight].to_numpy(float)
    size = f[survey.size].to_numpy(float)
    A_ref, A_alt = account_sources(reference), account_sources(alternative)

    # Survey income by source: several columns may share one.
    sources = sorted(set(survey.sources.values()))
    Y = pd.DataFrame(0.0, index=f.index, columns=sources)
    for col, s in survey.sources.items():
        Y[s] += f[col].to_numpy(float)

    recon_rows = []
    if reconcile:
        empty = [h for h in econ.hhs if not np.any(grp == h)]
        if empty:
            raise ValueError(f"household accounts {empty} have no survey household, so the "
                             f"survey cannot be reconciled with the SAM. Use reconcile=False, "
                             f"or a survey that covers every account.")
        problems = []
        for h in econ.hhs:
            m = grp == h
            for s in sources:
                if s == "fixed":
                    continue
                T = float(w[m] @ Y.loc[m, s].to_numpy())
                A = float(A_ref.at[h, s])
                if abs(T) <= 1e-12 and abs(A) <= 1e-12:
                    continue
                if abs(T) <= 1e-12 or abs(A) <= 1e-12 or T * A < 0:
                    problems.append((h, s, T, A))
                    continue
                Y.loc[m, s] *= A / T
                recon_rows.append(dict(group=h, source=s, survey=T, sam=A, factor=A / T))
            # A source the account has and the survey does not map at all.
            for s in _total_columns(econ):
                if s not in sources and abs(float(A_ref.at[h, s])) > 1e-9 and not any(
                        src.startswith(f"{s}:") for src in sources):
                    problems.append((h, s, 0.0, float(A_ref.at[h, s])))
        if problems:
            text = "; ".join(f"{h} {s}: survey {T:.6g}, SAM {A:.6g}" for h, s, T, A in problems)
            raise ValueError(
                f"the survey and the SAM disagree on which accounts have which income, so "
                f"scaling cannot reconcile them: {text}. Map the missing columns, or use "
                f"reconcile=False to apply relative changes only.")
    recon = pd.DataFrame(recon_rows, columns=["group", "source", "survey", "sam", "factor"])

    # Employment ratios of every segment, by account, from the model.
    segments = {}
    for mk_ref_name, mk_ref in (reference.labour_markets or {}).items():
        mk_alt = alternative.labour_markets[mk_ref_name]
        for k, fac in enumerate(mk_ref["segments"]):
            segments[fac] = (mk_ref["household_employment"][:, k],
                             mk_alt["household_employment"][:, k])
    seg_sources = [s for s in sources if s in segments]
    if reweight and any(":" in s and s.split(":")[0] in segments for s in sources):
        raise ValueError("a labour segment is mapped by activity; the reweighting moves "
                         "employment by segment, so map the segment whole")
    w_alt = w.copy()
    calib_rows = []
    reweighted = reweight and bool(seg_sources)
    if reweighted:
        U_ref = sum((m_["household_unemployed"] for m_ in reference.labour_markets.values()),
                    np.zeros(econ.n_hh))
        U_alt = sum((m_["household_unemployed"] for m_ in alternative.labour_markets.values()),
                    np.zeros(econ.n_hh))
        unemployed_cols = [c for c, v in (survey.workers or {}).items() if v == "unemployed"]
        for h in econ.hhs:
            m = grp == h
            if not m.any():
                continue
            hi = econ.hhs.index(h)
            cols, targets, names = [np.ones(int(m.sum()))], [float(w[m].sum())], ["households"]
            for s in sources:
                x = Y.loc[m, s].to_numpy()
                base = float(w[m] @ x)
                if abs(base) <= 1e-12:
                    continue
                if s in segments:
                    e0, e1 = segments[s][0][hi], segments[s][1][hi]
                    ratio = e1 / e0 if abs(e0) > 1e-12 else 1.0
                else:
                    ratio = 1.0
                cols.append(x)
                targets.append(base * ratio)
                names.append(s)
            if unemployed_cols and U_ref[hi] > 1e-12:
                x = f.loc[m, unemployed_cols].to_numpy(float).sum(axis=1)
                base = float(w[m] @ x)
                if base > 0:
                    cols.append(x)
                    targets.append(base * U_alt[hi] / U_ref[hi])
                    names.append("unemployed")
            X = np.column_stack(cols)
            new, iters = _rake(w[m], X, np.array(targets), names=names, group=h)
            w_alt[m] = new
            reached = X.T @ new
            base_tot = X.T @ w[m]
            for nm, b0, t, r in zip(names, base_tot, targets, reached):
                calib_rows.append(dict(group=h, constraint=nm, base=b0, target=t,
                                       reached=r, iterations=iters))
    calib = pd.DataFrame(calib_rows, columns=["group", "constraint", "base", "target",
                                              "reached", "iterations"])

    Y_alt = Y.copy()
    for h in econ.hhs:
        m = grp == h
        if not m.any():
            continue
        persons = float(w_alt[m] @ size[m])
        hi = econ.hhs.index(h)
        for s in sources:
            if s == "fixed":
                continue
            a0, a1 = float(A_ref.at[h, s]), float(A_alt.at[h, s])
            if abs(a0) > 1e-12:
                ratio = a1 / a0
                if reweighted and s in segments:
                    # The weights moved the employment; what is left is the
                    # change in income per worker.
                    e0, e1 = segments[s][0][hi], segments[s][1][hi]
                    ratio = ratio / (e1 / e0) if abs(e0) > 1e-12 and abs(e1) > 1e-12 else ratio
                Y_alt.loc[m, s] = Y.loc[m, s].to_numpy() * ratio
            elif abs(a1) > 1e-12:
                Y_alt.loc[m, s] = a1 * size[m] / persons
    # A source in neither the survey's mapping nor the reference, that the
    # alternative creates (a new transfer), is shared per person too.
    extra = pd.Series(0.0, index=f.index)
    for h in econ.hhs:
        m = grp == h
        if not m.any():
            continue
        persons = float(w_alt[m] @ size[m])
        for s in ("transfers", "remittances", "recycled"):
            if s in sources:
                continue
            a0, a1 = float(A_ref.at[h, s]), float(A_alt.at[h, s])
            if abs(a0) <= 1e-12 and abs(a1) > 1e-12:
                extra[m] += a1 * size[m] / persons
    if extra.any():
        Y_alt["new_flows"] = extra
        Y["new_flows"] = 0.0

    gross_ref = Y.sum(axis=1).to_numpy()
    gross_alt = Y_alt.sum(axis=1).to_numpy()
    rate_ref = _direct_rate(reference)
    rate_alt = _direct_rate(alternative)
    hidx = np.array([econ.hhs.index(h) for h in grp])
    disp_ref = gross_ref * (1.0 - rate_ref[hidx])
    disp_alt = gross_alt * (1.0 - rate_alt[hidx])

    shares = _budget_shares(survey, econ, hidx)
    rel = _effective_prices(alternative) / _effective_prices(reference)      # (n_hh, n_com)
    rel_i = rel[hidx]
    if price_index == "geometric":
        with np.errstate(divide="ignore"):
            P = np.exp(np.where(shares > 0, shares * np.log(rel_i), 0.0).sum(axis=1))
    else:
        P = (shares * rel_i).sum(axis=1)

    frame = pd.DataFrame({
        "group": grp, "weight": w, "weight_alt": w_alt, "size": size,
        "gross_ref": gross_ref, "gross_alt": gross_alt,
        "disposable_ref": disp_ref, "disposable_alt": disp_alt,
        "price_index": P, "real_ref": disp_ref, "real_alt": disp_alt / P}, index=f.index)
    rows = []
    for col, what in (survey.workers or {}).items():
        x = f[col].to_numpy(float)
        for h in econ.hhs:
            m = grp == h
            if m.any():
                rows.append(dict(group=h, column=col, workers=what,
                                 ref=float(w[m] @ x[m]), alt=float(w_alt[m] @ x[m])))
    workers = pd.DataFrame(rows, columns=["group", "column", "workers", "ref", "alt"])
    return MicroResult(frame=frame, income_ref=Y, income_alt=Y_alt, reconciliation=recon,
                       calibration=calib, workers=workers)


def _rake(w, X, targets, *, names=None, group=None, tol=1e-12, max_iter=200):
    """Calibrated weights by raking: ``w' = w exp(X λ)`` with ``X' w' = targets``.

    The raking distance keeps every weight positive and, when the constraints
    are cells that partition the sample, gives each cell its ratio
    ``target / total``. Newton on the multipliers, with the columns scaled by
    their base totals and a step halved until the gap falls; redundant
    constraints are handled by least squares. Returns the weights and the
    iterations it took.
    """
    w = np.asarray(w, float)
    X = np.asarray(X, float)
    t = np.asarray(targets, float)
    base = X.T @ w
    scale = np.where(np.abs(base) > 0, np.abs(base), np.maximum(np.abs(t), 1.0))
    Xs, ts = X / scale[None, :], t / scale
    lam = np.zeros(X.shape[1])

    def gap(lam_):
        u_ = w * np.exp(np.clip(Xs @ lam_, -700, 700))
        return u_, Xs.T @ u_ - ts

    u, F = gap(lam)
    for it in range(1, max_iter + 1):
        if np.max(np.abs(F)) < tol:
            return u, it - 1
        J = Xs.T @ (u[:, None] * Xs)
        step = np.linalg.lstsq(J, -F, rcond=None)[0]
        size_ = 1.0
        while size_ > 1e-10:
            u_new, F_new = gap(lam + size_ * step)
            if np.max(np.abs(F_new)) < np.max(np.abs(F)):
                break
            size_ /= 2.0
        else:
            break
        lam = lam + size_ * step
        u, F = u_new, F_new
    if np.max(np.abs(F)) < tol:
        return u, max_iter
    worst = int(np.argmax(np.abs(F)))
    label = names[worst] if names is not None else worst
    where = f" in account {group!r}" if group is not None else ""
    raise ValueError(
        f"the survey weights cannot be calibrated{where}: no positive weights reach the "
        f"target for {label!r} ({t[worst]:.6g}, base {base[worst]:.6g}) together with the "
        f"others. The survey's households in the account do not span the change the model "
        f"asks for; a larger survey, or reweight=False, is needed.")


def _direct_rate(period) -> np.ndarray:
    inc = np.asarray(period.hh_income, float)
    tax = (np.zeros_like(inc) if period.hh_direct_tax is None
           else np.asarray(period.hh_direct_tax, float))
    return np.where(np.abs(inc) > 0, tax / np.where(inc == 0, 1.0, inc), 0.0)


def _effective_prices(period) -> np.ndarray:
    """What each household account pays per unit of each commodity: wedges and carbon in."""
    from .decompose import _rates
    from .emissions import carbon_wedges
    r = _rates(period.econ, period)
    _, carb_hh = carbon_wedges(period)
    return period.pq[None, :] * (1.0 + r["W"] + r["Wg"]) + carb_hh[None, :]


def _budget_shares(survey, econ, hidx) -> np.ndarray:
    from .solve import _marginal_budget_shares
    account = _marginal_budget_shares(econ)[hidx]
    if survey.budget is None:
        return account
    spend = np.zeros((len(hidx), econ.n_com))
    for col, c in survey.budget.items():
        spend[:, econ.commodity(c)] += survey.frame[col].to_numpy(float)
    tot = spend.sum(axis=1, keepdims=True)
    # A household that reports no spending keeps its account's budget.
    return np.where(tot > 0, spend / np.where(tot > 0, tot, 1.0), account)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def inequality(result: MicroResult, *, equivalence: float = 1.0,
               epsilons: Sequence[float] = (0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Gini, Theil's T and L, GE(2) and Atkinson indices, in the two periods."""
    d = result.welfare(equivalence)
    rows = {}
    for col in ("ref", "alt"):
        x, p = d[col].to_numpy(), d[f"persons_{col}"].to_numpy()
        vals = {"gini": gini(x, p), "theil_t": theil(x, p), "theil_l": mean_log_deviation(x, p),
                "ge_2": generalized_entropy(x, 2.0, p)}
        for e in epsilons:
            vals[f"atkinson_{e:g}"] = atkinson(x, e, p)
        rows[col] = vals
    out = pd.DataFrame(rows)
    out["change"] = out["alt"] - out["ref"]
    return out


def poverty(result: MicroResult, lines, *, alphas: Sequence[int] = (0, 1, 2),
            by_group: bool = False, equivalence: float = 1.0) -> pd.DataFrame:
    """FGT indices for each poverty line and α, in the two periods.

    Lines are per adult equivalent, in the reference period's prices. With
    ``by_group`` the rows also hold each household account's contribution, and
    the contributions add up to the total.
    """
    d = result.welfare(equivalence)
    lines = [lines] if np.isscalar(lines) else list(lines)
    rows = []
    for z in lines:
        for a in alphas:
            row = dict(line=z, alpha=a)
            for col in ("ref", "alt"):
                persons = d[f"persons_{col}"]
                if by_group:
                    dec = fgt_decomposition(d[col], z, a, d["group"], persons)
                    for g in dec.index:
                        row[(col, g)] = float(dec.at[g, "contribution"])
                else:
                    row[col] = fgt(d[col], z, a, persons)
            rows.append(row)
    out = pd.DataFrame(rows).set_index(["line", "alpha"])
    if by_group:
        # Columns are (period, account), the last account being "total".
        out.columns = pd.MultiIndex.from_tuples(out.columns)
    else:
        out["change"] = out["alt"] - out["ref"]
    return out
