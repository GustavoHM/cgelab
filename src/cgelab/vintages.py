"""Putty-clay capital: an old vintage fixed in its sector, a new one that is not.

Capital that has been installed is clay: it stays in the activity that bought
it, and the labour it works with is fixed by the technology it was built for.
Capital not yet installed is putty: it goes where it earns most, with whatever
labour intensity the prices of the day make cheapest. Recursive models with
vintages, LINKAGE among them, carry the two as separate production bundles of
each activity.

Each activity's value added here is a CES of two bundles. The old bundle is a
CES of elasticity ``vintage_sigma`` (zero by default: fixed proportions, no
substitution ex post) over every factor but new capital; the new bundle is the
usual value-added CES of ``sigma_va`` (the substitution ex ante) over every
factor but old capital. Both bundles hold the activity's base-year factor
proportions, so at base prices each costs one and the base year is the model
without vintages. Old capital earns its own price in each activity (one pool
per activity, ω = 0, see :mod:`cgelab.mobility`); new capital earns one price.

LINKAGE makes the two bundles perfect substitutes. That case switches regime
when an activity shrinks faster than its old capital wears out, leaving old
capital idle, which a smooth solver does not do; here the bundles combine in a
CES of elasticity ``vintage_substitution``, 20 by default, and the distance to
perfect substitution falls as one over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .functions import ces_intensity, ces_price

__all__ = ["VintageNest"]


@dataclass
class VintageNest:
    """The calibrated two-bundle value added of a model with capital vintages."""

    old: str
    new: str
    i_old: int
    i_new: int
    share_old: np.ndarray     # (n_act,) old bundle's share of value added at base prices
    theta_old: np.ndarray     # (n_fac, n_act) factor shares in the old bundle
    theta_new: np.ndarray     # (n_fac, n_act) factor shares in the new bundle
    sigma_old: np.ndarray     # (n_act,) substitution ex post
    sigma_new: np.ndarray     # (n_act,) substitution ex ante, ``sigma_va``
    sigma_top: np.ndarray     # (n_act,) between the two bundles

    def prices(self, w: np.ndarray) -> tuple:
        """``(p_old, p_new, pva)`` per activity at the factor price matrix ``w``."""
        p_old = ces_price(self.theta_old, w, self.sigma_old)
        p_new = ces_price(self.theta_new, w, self.sigma_new)
        shares = np.vstack([self.share_old, 1.0 - self.share_old])
        pva = ces_price(shares, np.vstack([p_old, p_new]), self.sigma_top)
        return p_old, p_new, pva

    def bundles(self, w: np.ndarray) -> np.ndarray:
        """Each bundle per unit of value added, ``(2, n_act)``: old, then new."""
        p_old, p_new, pva = self.prices(w)
        shares = np.vstack([self.share_old, 1.0 - self.share_old])
        return ces_intensity(shares, np.vstack([p_old, p_new]), pva, self.sigma_top)

    def installed(self, w: np.ndarray, bundles: np.ndarray) -> "VintageNest":
        """The nest of the year after: this year's new bundle has joined the old one.

        Clay keeps the technique it was installed with. The old bundle's inputs
        per unit become the average, weighted by the two bundles' quantities
        (``bundles``, ``(2, n_act)``), of its own and of the new bundle's at the
        prices ``w`` it was installed at, new capital becoming old capital; its
        CES is then recalibrated to use those inputs at those prices. Both
        bundles wear out at the rate capital does, so the weights need no
        depreciation. With the same elasticity in both bundles, and old and new
        capital at one price, the new technique is the old one and nothing
        changes.
        """
        import dataclasses
        p_old, p_new, _ = self.prices(w)
        q_old = ces_intensity(self.theta_old, w, p_old, self.sigma_old)
        q_new = ces_intensity(self.theta_new, w, p_new, self.sigma_new)
        q_new[self.i_old], q_new[self.i_new] = q_new[self.i_new], 0.0
        b_old, b_new = np.asarray(bundles[0], float), np.asarray(bundles[1], float)
        total = b_old + b_new
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(total > 0, (q_old * b_old + q_new * b_new) / total, q_old)
        cost = (q * w).sum(axis=0)
        theta = np.where(q > 0, q * (w / cost[None, :]) ** self.sigma_old[None, :], 0.0)
        return dataclasses.replace(self, theta_old=theta)

    def intensity(self, w: np.ndarray) -> np.ndarray:
        """Factor input per unit of value added, ``(n_fac, n_act)``."""
        p_old, p_new, _ = self.prices(w)
        b = self.bundles(w)
        return (ces_intensity(self.theta_old, w, p_old, self.sigma_old) * b[0][None, :]
                + ces_intensity(self.theta_new, w, p_new, self.sigma_new) * b[1][None, :])


def build_vintages(spec, facs, acts, theta, leontief, sigma_va):
    """The vintage nest of ``spec.vintages``, from the calibrated value-added shares."""
    if not spec.vintages:
        return None
    (old, new), = spec.vintages.items()
    i_old, i_new = facs.index(old), facs.index(new)
    capital = theta[i_old] + theta[i_new]
    uses = capital > 0
    fixed = [a for a, u, lt in zip(acts, uses, leontief) if u and lt]
    if fixed:
        raise ValueError(
            f"activities {fixed} have fixed-proportions value added (negative value added "
            f"in the SAM, or value_added='leontief'), and vintages need the value-added "
            f"CES. Treat the negative value added with negative_value_added='net'.")
    one = [a for a, u, o, n in zip(acts, uses, theta[i_old], theta[i_new])
           if u and (o <= 0 or n <= 0)]
    if one:
        raise ValueError(
            f"activities {one} hold capital of one vintage only. Each activity that uses "
            f"capital needs both: the new capital it installs this year is its old capital "
            f"the next. Give split_vintages a share strictly between 0 and 1.")
    share_old = np.where(uses, theta[i_old] / np.where(uses, capital, 1.0), 0.0)
    theta_old, theta_new = theta.copy(), theta.copy()
    theta_old[i_old], theta_old[i_new] = capital, 0.0
    theta_new[i_new], theta_new[i_old] = capital, 0.0
    vec = spec.vintage_sigma_vector(acts), spec.vintage_substitution_vector(acts)
    return VintageNest(old=old, new=new, i_old=i_old, i_new=i_new, share_old=share_old,
                       theta_old=theta_old, theta_new=theta_new, sigma_old=vec[0],
                       sigma_new=np.asarray(sigma_va, dtype=float), sigma_top=vec[1])
