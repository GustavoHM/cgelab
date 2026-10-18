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

__all__ = ["ces_price", "ces_intensity", "leontief_price", "armington", "cet", "cet_domestic_price",
           "cobb_douglas_price"]


def _by_column(prices: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Input prices as ``(n_inputs, n_columns)``.

    A vector is one price per input, the same in every column. A matrix is a
    price per input and column: a factor that is not perfectly mobile earns a
    different price in each activity, and the nest of each activity sees its own.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 1:
        return np.broadcast_to(prices[:, None], shares.shape)
    if prices.shape != shares.shape:
        raise ValueError(f"prices of shape {prices.shape} do not match shares {shares.shape}")
    return prices


def cobb_douglas_price(shares: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Unit cost of a Cobb-Douglas nest: ``prod_i p_i^{theta_i}``, by column.

    ``prices`` is one per input, or ``(n_inputs, n_columns)``.
    """
    prices = _by_column(prices, shares)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.prod(np.where(shares > 0, prices ** shares, 1.0), axis=0)


def ces_price(shares: np.ndarray, prices: np.ndarray, sigma) -> np.ndarray:
    """Unit cost of a CES nest, column by column.

    ``shares`` is ``(n_inputs, n_columns)`` and sums to one down each column;
    ``prices`` is one per input, or a matrix of the same shape as ``shares``;
    ``sigma`` is a scalar or one value per column. ``sigma == 1`` falls back to
    Cobb-Douglas, which is its limit and not a special case to be approximated.
    """
    prices = _by_column(prices, shares)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(shares.shape[1], float(sigma))
    out = np.zeros(shares.shape[1])
    unit = np.abs(sigma - 1.0) < 1e-9
    if unit.any():
        out[unit] = cobb_douglas_price(shares[:, unit], prices[:, unit])
    other = ~unit
    if other.any():
        s = sigma[other]
        sh = shares[:, other]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(sh > 0, sh * prices[:, other] ** (1.0 - s[None, :]), 0.0)
            tot = terms.sum(axis=0)
            out[other] = np.where(sh.sum(axis=0) > 0, tot ** (1.0 / (1.0 - s)), 0.0)
    return out


def ces_intensity(shares: np.ndarray, prices: np.ndarray, unit_cost: np.ndarray,
                  sigma) -> np.ndarray:
    """Input per unit of the CES composite: ``theta_i (P/p_i)^sigma``.

    ``prices`` is one per input, or a matrix of the same shape as ``shares``.
    """
    prices = _by_column(prices, shares)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(shares.shape[1], float(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(shares > 0,
                        shares * (unit_cost[None, :] / prices) ** sigma[None, :], 0.0)


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


def cet_domestic_price(export_share: np.ndarray, p_output: np.ndarray, p_export: np.ndarray,
                       sigma) -> np.ndarray:
    """Domestic sales price that makes the CET revenue function equal ``p_output``.

    The inverse of :func:`cet` in its domestic-price argument. Under constant
    returns the producer price of a commodity is its unit cost, so what the
    transformation frontier determines is not that price but how it splits:
    given the output price and the export price, the domestic price is the one
    at which ``pz(p_domestic, p_export) = p_output``.

    Degenerate cases return finite numbers: a commodity with no exports sells at
    its output price, and one that is exported whole has no domestic sale, so its
    domestic price is set to the output price as well (it multiplies zero).
    When the export price is so high that no positive domestic price satisfies
    the frontier, the domestic price is floored near zero and domestic sales
    vanish -- the economic content of that corner.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(export_share.shape, float(sigma))
    se = export_share
    sd = 1.0 - se
    e = 1.0 + sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = (p_output ** e - se * p_export ** e) / np.where(sd > 0, sd, 1.0)
        inner = np.maximum(inner, 1e-300)
        pd = inner ** (1.0 / e)
    interior = (se > 0) & (sd > 0)
    return np.where(interior, pd, p_output)
