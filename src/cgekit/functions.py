"""Price and quantity blocks of the standard nests.

Each function takes calibrated shares and prices and returns the unit cost index
and the implied input intensities, so the solver reads as the model's algebra
rather than as its arithmetic.

Every nest is written to survive the degenerate cases a real SAM contains: an
activity with no value added, a commodity with no imports, a share vector that
sums to zero. Those return finite numbers rather than NaN, because one NaN in a
131-commodity system propagates everywhere and the traceback points at the
solver instead of at the account that caused it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ces_price", "ces_intensity", "leontief_price", "armington", "cet",
           "cobb_douglas_price"]


def cobb_douglas_price(shares: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Unit cost of a Cobb-Douglas nest: ``prod_i p_i^{theta_i}``, by column."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.prod(np.where(shares > 0, prices[:, None] ** shares, 1.0), axis=0)


def ces_price(shares: np.ndarray, prices: np.ndarray, sigma) -> np.ndarray:
    """Unit cost of a CES nest, column by column.

    ``shares`` is ``(n_inputs, n_columns)`` and sums to one down each column;
    ``sigma`` is a scalar or one value per column. ``sigma == 1`` falls back to
    Cobb-Douglas, which is its limit and not a special case to be approximated.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(shares.shape[1], float(sigma))
    out = np.zeros(shares.shape[1])
    unit = np.abs(sigma - 1.0) < 1e-9
    if unit.any():
        out[unit] = cobb_douglas_price(shares[:, unit], prices)
    other = ~unit
    if other.any():
        s = sigma[other]
        sh = shares[:, other]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(sh > 0, sh * prices[:, None] ** (1.0 - s[None, :]), 0.0)
            tot = terms.sum(axis=0)
            out[other] = np.where(sh.sum(axis=0) > 0, tot ** (1.0 / (1.0 - s)), 0.0)
    return out


def ces_intensity(shares: np.ndarray, prices: np.ndarray, unit_cost: np.ndarray,
                  sigma) -> np.ndarray:
    """Input per unit of the CES composite: ``theta_i (P/p_i)^sigma``."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(shares.shape[1], float(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(shares > 0,
                        shares * (unit_cost[None, :] / prices[:, None]) ** sigma[None, :], 0.0)


def leontief_price(unit_inputs: np.ndarray, prices: np.ndarray,
                   composite_per_unit: np.ndarray) -> np.ndarray:
    """Unit cost of a fixed-proportions nest, expressed per unit of the composite."""
    cost = prices @ unit_inputs
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(composite_per_unit != 0,
                        cost / np.where(composite_per_unit == 0, 1.0, composite_per_unit), 0.0)


def armington(domestic_share: np.ndarray, p_domestic: np.ndarray, p_import: np.ndarray,
              sigma) -> tuple:
    """CES between the domestic and the imported variety of each commodity.

    Returns ``(composite price, domestic intensity, import intensity)``, both
    intensities per unit of the composite.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(domestic_share.shape, float(sigma))
    sd = domestic_share
    sm = 1.0 - sd
    with np.errstate(divide="ignore", invalid="ignore"):
        pq = (sd * p_domestic ** (1 - sigma) + sm * p_import ** (1 - sigma)) ** (1 / (1 - sigma))
        dsh = np.where(p_domestic > 0, sd * (pq / p_domestic) ** sigma, 0.0)
        msh = np.where(p_import > 0, sm * (pq / p_import) ** sigma, 0.0)
    return pq, dsh, msh


def cet(export_share: np.ndarray, p_domestic: np.ndarray, p_export: np.ndarray,
        sigma) -> tuple:
    """Constant elasticity of transformation between domestic sales and exports.

    A caution worth repeating where it will be read: the CET elasticity is the
    one parameter of a standard CGE that input-output sources do not supply. If
    no headline result may rest on an invented number, the honest baseline fixes
    export shares (``ModelSpec.export_rule = "fixed_shares"``) and brings the CET
    in only as a robustness variant.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(export_share.shape, float(sigma))
    se = export_share
    sd = 1.0 - se
    with np.errstate(divide="ignore", invalid="ignore"):
        pz = (sd * p_domestic ** (1 + sigma) + se * p_export ** (1 + sigma)) ** (1 / (1 + sigma))
        dsh = np.where(pz > 0, sd * (p_domestic / pz) ** sigma, 0.0)
        esh = np.where(pz > 0, se * (p_export / pz) ** sigma, 0.0)
    return pz, dsh, esh
