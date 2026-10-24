"""Behavioural microsimulation, in the manner of Bourguignon, Robilliard and Robinson (2005).

The top-down link in :mod:`cgelab.micro` moves income source by source and,
under labour segments, moves employment through the survey weights. Here the
people decide. Each person of working age has an occupational status (a labour
segment, ``"unemployed"``, ``"inactive"``) chosen by a multinomial logit on
their characteristics, and earns, in a segment, what an earnings equation
estimated on that segment predicts, plus their own residual. A counterfactual
reaches them through two sets of intercepts, by SAM account: the choice
intercepts move until employment in each segment changes as the model says, and
the earnings intercepts until earnings per worker change as the model's wages
do.

The residuals of the choice are what the observed choice reveals: draws of the
Gumbel errors conditional on the status each person was observed in, so that
the base reproduces the survey. Two ways to use them:

* ``"draws"`` -- BRR's procedure. Each replication draws the residuals, and a
  person moves where the shifted utilities put them. Results depend on the seed;
  the replications give a Monte Carlo mean and its standard error.
* ``"expected"`` -- the limit of those replications, in closed form. With
  intercepts shifted by ``δ``, the probability that a person observed in ``j``
  chooses ``k`` is ``T[j, k] / P[j]``, where ``T`` is the joint probability of
  the two choices under the logit. ``T[j, j]`` is a logit with shifted
  utilities; ``T[j, k]`` for ``k ≠ j`` is a one-dimensional integral that is
  piecewise analytic. It is deterministic and smooth in ``δ``, which is what
  lets the intercepts hit the model's targets exactly.

Movers into a segment earn the equation's prediction: in ``"expected"`` mode
times ``exp(σ²/2)``, the mean of the log-normal error; in ``"draws"`` mode with
a normal draw. Stayers keep their own residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["PersonSurvey", "Behaviour", "transition_probabilities", "conditional_draws"]


@dataclass
class PersonSurvey:
    """A survey of persons: their household, status, labour earnings and characteristics.

    ``household`` holds the index of the person's row in ``Survey.frame``.
    ``status`` is a labour segment of the model, ``"unemployed"``, ``"inactive"``,
    or any other non-working status. ``characteristics`` are the regressors of
    the earnings equations and, unless ``choice_characteristics`` says
    otherwise, of the occupational choice. Persons outside the labour force by
    age belong in the household survey's other sources, not here.
    """

    frame: pd.DataFrame
    household: str = "household"
    status: str = "status"
    earnings: str = "earnings"
    characteristics: Sequence[str] = ()
    choice_characteristics: Sequence[str] | None = None

    def __post_init__(self):
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("PersonSurvey.frame must be a DataFrame, one row per person")
        cols = [self.household, self.status, self.earnings, *self.characteristics,
                *(self.choice_characteristics or ())]
        missing = [c for c in cols if c not in self.frame.columns]
        if missing:
            raise KeyError(f"columns not in the person survey: {missing}")
        y = self.frame[self.earnings].to_numpy(float)
        if not (np.all(np.isfinite(y)) and np.all(y >= 0)):
            raise ValueError("earnings must be finite and non-negative")
        for c in cols[3:]:
            if not np.all(np.isfinite(self.frame[c].to_numpy(float))):
                raise ValueError(f"characteristic {c!r} has missing or infinite values")

    @property
    def choice_columns(self) -> list:
        return list(self.characteristics if self.choice_characteristics is None
                    else self.choice_characteristics)

    def design(self, columns) -> np.ndarray:
        return np.column_stack([np.ones(len(self.frame))]
                               + [self.frame[c].to_numpy(float) for c in columns])


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

@dataclass
class Behaviour:
    """An estimated occupational choice and earnings equations.

    ``statuses`` lists the choices, ``base`` among them; ``choice`` holds one
    row of coefficients per status (intercept first, then
    ``PersonSurvey.choice_columns``), zero for the base. ``earnings`` maps each
    segment to ``{"coef", "sigma", "n"}``, the coefficients of ``ln y`` on an
    intercept and ``PersonSurvey.characteristics``.
    """

    statuses: list
    base: str
    segments: list
    choice: np.ndarray
    earnings: dict
    loglik: float = float("nan")
    iterations: int = 0

    @classmethod
    def estimate(cls, persons: PersonSurvey, segments: Sequence[str], *, weights=None,
                 base: str | None = None) -> "Behaviour":
        """Fit the earnings equations by least squares and the choice by maximum likelihood.

        ``weights`` (one per person, the survey weights) make both
        pseudo-likelihood estimates of the population. ``base`` is the status
        whose coefficients are zero: ``"inactive"`` when there is one, else the
        last non-working status, else the last segment.
        """
        from scipy.optimize import minimize

        f = persons.frame
        status = f[persons.status].astype(str).to_numpy()
        segments = list(segments)
        others = sorted(set(status) - set(segments))
        statuses = segments + others
        missing = [s for s in segments if s not in set(status)]
        if missing:
            raise ValueError(f"no person works in segments {missing}; their equations cannot "
                             f"be estimated")
        if base is None:
            base = "inactive" if "inactive" in others else (others[-1] if others
                                                             else segments[-1])
        if base not in statuses:
            raise ValueError(f"base status {base!r} is not among {statuses}")
        w = (np.ones(len(f)) if weights is None else np.asarray(weights, float))
        if w.shape != (len(f),) or np.any(w <= 0):
            raise ValueError("weights must be positive, one per person")
        y = f[persons.earnings].to_numpy(float)

        X = persons.design(persons.characteristics)
        earnings = {}
        for s in segments:
            m = (status == s) & (y > 0)
            if m.sum() < X.shape[1]:
                raise ValueError(f"segment {s!r} has {int(m.sum())} earners with positive "
                                 f"earnings, fewer than the {X.shape[1]} coefficients")
            sw = np.sqrt(w[m])
            coef = np.linalg.lstsq(X[m] * sw[:, None], np.log(y[m]) * sw, rcond=None)[0]
            resid = np.log(y[m]) - X[m] @ coef
            sigma = float(np.sqrt(w[m] @ resid ** 2 / w[m].sum()))
            earnings[s] = dict(coef=coef, sigma=sigma, n=int(m.sum()))

        Z = persons.design(persons.choice_columns)
        K, q = len(statuses), Z.shape[1]
        free = [k for k, s in enumerate(statuses) if s != base]
        Y = (status[:, None] == np.array(statuses)[None, :]).astype(float)

        def unpack(theta):
            B = np.zeros((K, q))
            B[free] = theta.reshape(len(free), q)
            return B

        def nll(theta):
            V = Z @ unpack(theta).T
            top = V.max(axis=1, keepdims=True)
            lse = top[:, 0] + np.log(np.exp(V - top).sum(axis=1))
            ll = w @ ((Y * V).sum(axis=1) - lse)
            P = np.exp(V - lse[:, None])
            grad = ((Y - P) * w[:, None]).T @ Z                 # (K, q)
            return -ll, -grad[free].ravel()

        sol = minimize(nll, np.zeros(len(free) * q), jac=True, method="BFGS",
                       options={"gtol": 1e-10, "maxiter": 5000})
        return cls(statuses=statuses, base=base, segments=segments, choice=unpack(sol.x),
                   earnings=earnings, loglik=-float(sol.fun), iterations=int(sol.nit))

    def utilities(self, persons: PersonSurvey) -> np.ndarray:
        """Deterministic utilities, ``(n_persons, n_statuses)``."""
        return persons.design(persons.choice_columns) @ self.choice.T

    def predicted_log_earnings(self, persons: PersonSurvey) -> np.ndarray:
        """``ln y`` predicted in every segment, ``(n_persons, n_segments)``."""
        X = persons.design(persons.characteristics)
        return np.column_stack([X @ self.earnings[s]["coef"] for s in self.segments])


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def transition_probabilities(V: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Joint probabilities of the choice before and after shifting utilities by ``delta``.

    ``V`` is ``(n, K)``, ``delta`` ``(K,)``, the same for every person. Returns
    ``T`` of shape ``(n, K, K)`` with ``T[i, j, k]`` the probability that person
    ``i`` chooses ``j`` with utilities ``V`` and ``k`` with ``V + delta``, under
    i.i.d. Gumbel errors that do not change. Its rows add up to the logit
    probabilities of the first choice and its columns to those of the second.

    ``T[j, j]`` is a logit in which every other alternative ``l`` is handicapped
    by ``max(0, δ_l − δ_j)``. For ``k ≠ j``, with ``a = exp(V)`` and ``D = δ_k −
    δ_j > 0``, ``T[j, k] = ∫_0^D a_j a_k e^d / (α(d) + β(d) e^d)² dd``, where an
    alternative ``l`` enters ``α`` with ``a_l`` while ``δ_l ≤ δ_k − d`` and ``β``
    with ``a_l e^{δ_l − δ_k}`` after; between those breakpoints the integral is
    ``a_j a_k / β · [1/(α + β e^{d0}) − 1/(α + β e^{d1})]``.
    """
    V = np.asarray(V, float)
    d = np.asarray(delta, float)
    n, K = V.shape
    a = np.exp(V - V.max(axis=1, keepdims=True))       # scale-free: T is homogeneous
    T = np.zeros((n, K, K))
    for j in range(K):
        for k in range(K):
            if k == j:
                handicap = np.exp(np.maximum(0.0, d - d[j]))
                handicap[j] = 1.0
                T[:, j, j] = a[:, j] / (a * handicap[None, :]).sum(axis=1)
                continue
            D = d[k] - d[j]
            if D <= 0:
                continue
            rest = [l for l in range(K) if l not in (j, k)]
            cuts = sorted({0.0, D} | {d[k] - d[l] for l in rest if 0.0 < d[k] - d[l] < D})
            for d0, d1 in zip(cuts[:-1], cuts[1:]):
                mid = 0.5 * (d0 + d1)
                early = [l for l in rest if d[l] <= d[k] - mid]
                late = [l for l in rest if d[l] > d[k] - mid]
                alpha = a[:, j] + (a[:, early].sum(axis=1) if early else 0.0)
                beta = a[:, k] + ((a[:, late] * np.exp(d[late] - d[k])[None, :]).sum(axis=1)
                                  if late else 0.0)
                T[:, j, k] += (a[:, j] * a[:, k] / beta
                               * (1.0 / (alpha + beta * np.exp(d0))
                                  - 1.0 / (alpha + beta * np.exp(d1))))
    return T


def link(persons: PersonSurvey, behaviour: Behaviour, *, person_group, person_weight,
         household_of, n_households, hhs, A_ref, A_alt, reference, alternative,
         reconcile=True, method="expected", seed=0, replications=100) -> dict:
    """Carry the model's labour market to the persons; the household labour income it gives.

    Per SAM account: the choice intercepts ``δ`` that move employment in each
    segment (and unemployment, where the model has a queue) as the account's
    does, and the earnings intercepts that move earnings per worker as the
    account's income per worker does. Returns the households' labour income by
    segment in the reference and the alternative, the replications in
    ``"draws"`` mode, and the calibration and head-count tables.
    """
    from scipy.optimize import root

    if method not in ("expected", "draws"):
        raise ValueError(f"method must be 'expected' or 'draws', got {method!r}")
    f = persons.frame
    statuses, segs = behaviour.statuses, behaviour.segments
    status = f[persons.status].astype(str).to_numpy()
    unknown = sorted(set(status) - set(statuses))
    if unknown:
        raise ValueError(f"statuses {unknown} were not in the estimation ({statuses})")
    chosen = np.array([statuses.index(s) for s in status])
    w = np.asarray(person_weight, float)
    y = f[persons.earnings].to_numpy(float)
    V = behaviour.utilities(persons)
    lnpred = behaviour.predicted_log_earnings(persons)                  # (n, n_seg)
    sigma = np.array([behaviour.earnings[s]["sigma"] for s in segs])

    # Employment and income per worker, by account and segment, from the model.
    emp = {}
    for mk_name, mk in (reference.labour_markets or {}).items():
        mk1 = alternative.labour_markets[mk_name]
        for k, fac in enumerate(mk["segments"]):
            emp[fac] = (mk["household_employment"][:, k], mk1["household_employment"][:, k])
    missing = [s for s in segs if s not in emp]
    if missing:
        raise ValueError(f"segments {missing} are not labour segments of the model")
    U0 = sum((m_["household_unemployed"] for m_ in reference.labour_markets.values()),
             np.zeros(len(hhs)))
    U1 = sum((m_["household_unemployed"] for m_ in alternative.labour_markets.values()),
             np.zeros(len(hhs)))

    kappa = np.ones((len(hhs), len(segs)))
    recon_rows, calib_rows, worker_rows, problems = [], [], [], []
    base = np.zeros((n_households, len(segs)))
    alt = np.zeros((n_households, len(segs)))
    draws = (np.zeros((replications, n_households, len(segs))) if method == "draws" else None)
    rng = np.random.default_rng(seed)

    for hi, h in enumerate(hhs):
        m = person_group == h
        if not m.any():
            continue
        # Reconcile each segment's earnings with the account's income from it.
        for k, s in enumerate(segs):
            inside = m & (chosen == statuses.index(s))
            T_ = float(w[inside] @ y[inside])
            A_ = float(A_ref.at[h, s])
            if not reconcile or (abs(T_) <= 1e-12 and abs(A_) <= 1e-12):
                continue
            if abs(T_) <= 1e-12 or abs(A_) <= 1e-12:
                problems.append((h, s, T_, A_))
                continue
            kappa[hi, k] = A_ / T_
            recon_rows.append(dict(group=h, source=s, survey=T_, sam=A_, factor=A_ / T_))
        yb = y[m].copy()
        for k, s in enumerate(segs):
            yb[chosen[m] == statuses.index(s)] *= kappa[hi, k]
        Vh, ch, wh = V[m], chosen[m], w[m]
        P0 = np.exp(Vh - Vh.max(axis=1, keepdims=True))
        P0 /= P0.sum(axis=1, keepdims=True)
        rows = np.arange(len(ch))

        # Targets: persons in each segment, and the unemployed where the model has a queue.
        targets = {}
        for k, s in enumerate(segs):
            e0, e1 = emp[s][0][hi], emp[s][1][hi]
            E0 = float(wh[ch == statuses.index(s)].sum())
            if E0 > 0 and abs(e0) > 1e-12 and s != behaviour.base:
                targets[statuses.index(s)] = E0 * e1 / e0
        if "unemployed" in statuses and U0[hi] > 1e-12 and behaviour.base != "unemployed":
            u = statuses.index("unemployed")
            E0 = float(wh[ch == u].sum())
            if E0 > 0:
                targets[u] = E0 * U1[hi] / U0[hi]
        free = sorted(targets)

        def expected(delta_free):
            delta = np.zeros(len(statuses))
            delta[free] = delta_free
            T = transition_probabilities(Vh, delta)
            return T[rows, ch] / P0[rows, ch][:, None], delta     # π: (n_h, K)

        def gap(delta_free):
            pi, _ = expected(delta_free)
            got = wh @ pi
            return np.array([np.log(got[k] / targets[k]) for k in free])

        iters = 0
        if free:
            sol = root(gap, np.zeros(len(free)), method="hybr", options={"xtol": 1e-14})
            iters = int(sol.nfev)
            if np.max(np.abs(gap(sol.x))) > 1e-10:
                raise ValueError(
                    f"account {h!r}: no shift of the choice intercepts moves employment as the "
                    f"model does (largest gap {np.max(np.abs(gap(sol.x))):.3g} in logs). The "
                    f"survey's persons in the account cannot reach the targets.")
            delta_free = sol.x
        else:
            delta_free = np.zeros(0)
        pi, delta = expected(delta_free)
        got = wh @ pi
        for k_ in free:
            E0 = float(wh[ch == k_].sum())
            calib_rows.append(dict(group=h, constraint=f"persons {statuses[k_]}", base=E0,
                                   target=targets[k_], reached=float(got[k_]),
                                   iterations=iters))
        for k_, s in enumerate(statuses):
            worker_rows.append(dict(group=h, column=s, workers=s,
                                    ref=float(wh[ch == k_].sum()), alt=float(got[k_])))

        # Earnings: stayers keep their own, movers earn the prediction.
        shift = np.zeros(len(segs))
        hh = household_of[m]
        for k, s in enumerate(segs):
            ks = statuses.index(s)
            stay = ch == ks
            mover = np.exp(lnpred[m, k] + 0.5 * sigma[k] ** 2) * kappa[hi, k]
            yhat = np.where(stay, yb, mover)
            E0 = float(wh[stay].sum())
            if E0 <= 0:
                continue
            e0, e1 = emp[s][0][hi], emp[s][1][hi]
            a0, a1 = float(A_ref.at[h, s]), float(A_alt.at[h, s])
            per_worker = (a1 / a0) / (e1 / e0) if abs(a0) > 1e-12 and abs(e1) > 1e-12 else 1.0
            mean0 = float(wh[stay] @ yb[stay]) / E0
            mass = float(wh @ pi[:, ks])
            mean_now = float(wh @ (pi[:, ks] * yhat)) / mass if mass > 0 else mean0
            shift[k] = np.log(mean0 * per_worker / mean_now) if mean_now > 0 else 0.0
            np.add.at(base[:, k], hh[stay], yb[stay])
            np.add.at(alt[:, k], hh, pi[:, ks] * yhat * np.exp(shift[k]))
            calib_rows.append(dict(group=h, constraint=f"earnings {s}", base=mean0,
                                   target=mean0 * per_worker,
                                   reached=mean_now * np.exp(shift[k]), iterations=0))
        if draws is not None:
            for r in range(replications):
                eps = conditional_draws(Vh, ch, rng)
                new = np.argmax(Vh + delta[None, :] + eps, axis=1)
                z = rng.standard_normal((len(ch), len(segs)))
                for k, s in enumerate(segs):
                    ks = statuses.index(s)
                    there = new == ks
                    yk = np.where(ch == ks, yb,
                                  np.exp(lnpred[m, k] + sigma[k] * z[:, k]) * kappa[hi, k])
                    np.add.at(draws[r, :, k], hh[there], yk[there] * np.exp(shift[k]))
    if problems:
        text = "; ".join(f"{h} {s}: survey {T_:.6g}, SAM {A_:.6g}" for h, s, T_, A_ in problems)
        raise ValueError(
            f"the person survey and the SAM disagree on which accounts earn in which segment: "
            f"{text}. Use reconcile=False to apply relative changes only.")
    return dict(base=base, alt=alt, draws=draws, recon=recon_rows, calib=calib_rows,
                workers=worker_rows)


def conditional_draws(V: np.ndarray, chosen: np.ndarray, rng) -> np.ndarray:
    """Gumbel errors drawn conditional on ``chosen`` being each person's best choice.

    The maximum utility is drawn first, ``ln Σ exp(V) + G``; the chosen
    alternative's error makes its utility that maximum, and every other
    alternative's error is drawn from its Gumbel truncated below it.
    """
    V = np.asarray(V, float)
    n, K = V.shape
    top = V.max(axis=1, keepdims=True)
    M = top[:, 0] + np.log(np.exp(V - top).sum(axis=1)) - np.log(-np.log(rng.uniform(size=n)))
    eps = np.empty_like(V)
    F_M = np.exp(-np.exp(-(M[:, None] - V)))              # each Gumbel's CDF at the maximum
    u = rng.uniform(size=(n, K)) * F_M
    eps[:] = -np.log(-np.log(u))
    rows = np.arange(n)
    eps[rows, chosen] = M - V[rows, chosen]
    return eps
