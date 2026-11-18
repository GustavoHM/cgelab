"""A symmetric two-country, one-sector Melitz model with Pareto productivity.

One sector and one factor, labour, in each of two identical countries. Entrants pay an entry
cost, draw productivity from a Pareto distribution and serve the domestic market if its
operating profit covers the domestic fixed cost. Firms above a second cutoff also export, paying
an export fixed cost and an iceberg trade cost. CES monopolistic competition gives every firm the
markup ``sigma / (sigma - 1)``; free entry exhausts expected profits. Labour is the numeraire,
trade is balanced by symmetry and welfare is real CES consumption.

The model is static, as in the textbook version of Melitz (2003): there is no death shock, so
``entry_cost`` is the per-period equivalent of the sunk cost (his ``delta * f_e``).

The Pareto distribution makes the two cutoffs, the firm masses and the price index available in
closed form, and that is all this module computes. It is a standalone model: it reads no SAM, no
``ModelSpec`` and no ``Scenario``, and ``solve_period`` knows nothing of it. Firm heterogeneity
as a sectoral block of the SAM-based equilibrium (Balistreri and Rutherford; Dixon, Jerie and
Rimmer) is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["MelitzSpec", "MelitzResult", "solve_melitz"]


@dataclass(frozen=True)
class MelitzSpec:
    """Parameters of the symmetric one-sector Melitz model.

    Fixed and entry costs are amounts of labour, each paid in the firm's own country.
    ``pareto_shape`` must exceed ``sigma - 1`` so that expected revenue and profits are finite.
    The solved case has interior selection:
    ``iceberg_cost ** (sigma - 1) * export_fixed_cost / domestic_fixed_cost > 1`` and the
    domestic cutoff is above ``productivity_min``.
    """

    sigma: float = 4.0
    pareto_shape: float = 5.0
    productivity_min: float = 1.0
    labour: float = 100.0
    entry_cost: float = 1.0
    domestic_fixed_cost: float = 1.0
    export_fixed_cost: float = 1.0
    iceberg_cost: float = 1.5


@dataclass(frozen=True)
class MelitzResult:
    """The equilibrium of one country; the other country is identical.

    ``entrants``, ``domestic_firms`` and ``exporters`` are masses of firms. Revenues are
    aggregates in units of labour. ``average_domestic_productivity`` and
    ``average_export_productivity`` are Melitz's averages over the firms that serve each market:
    the productivity of the representative firm that reproduces the CES aggregate,
    ``E[phi ** (sigma - 1)] ** (1 / (sigma - 1))``. ``residuals`` are the two zero-profit
    conditions evaluated with the price index, and the free-entry, labour and revenue identities;
    ``residual`` is the largest in absolute value.
    """

    spec: MelitzSpec
    domestic_cutoff: float
    export_cutoff: float
    entrants: float
    domestic_firms: float
    exporters: float
    price_index: float
    welfare: float
    domestic_revenue: float
    export_revenue: float
    average_domestic_productivity: float
    average_export_productivity: float
    residual: float
    residuals: dict[str, float] = field(repr=False)

    @property
    def real_wage(self) -> float:
        """Labour's real return, with the nominal wage normalized to one."""
        return 1.0 / self.price_index

    @property
    def export_share(self) -> float:
        """The share of active domestic firms that export."""
        return self.exporters / self.domestic_firms

    @property
    def entry_probability(self) -> float:
        """The probability that an entrant draws productivity above the domestic cutoff."""
        return self.domestic_firms / self.entrants

    @property
    def domestic_share(self) -> float:
        """The share of expenditure that falls on varieties produced at home."""
        return self.domestic_revenue / (self.domestic_revenue + self.export_revenue)

    @property
    def varieties(self) -> float:
        """The mass of varieties a consumer can buy: domestic firms and the partner's exporters."""
        return self.domestic_firms + self.exporters

    def revenue(self, productivity, *, market: str = "domestic"):
        """Revenue at ``productivity`` in the ``domestic`` or ``export`` market."""
        if market not in ("domestic", "export"):
            raise ValueError("market is 'domestic' or 'export'")
        phi = np.asarray(productivity, dtype=float)
        if np.any(~np.isfinite(phi)) or np.any(phi <= 0):
            raise ValueError("productivity must be finite and positive")
        s = self.spec
        trade = s.iceberg_cost ** (1.0 - s.sigma) if market == "export" else 1.0
        rho = (s.sigma - 1.0) / s.sigma
        return s.labour * self.price_index ** (s.sigma - 1.0) * (rho * phi) ** (s.sigma - 1.0) * trade

    def operating_profit(self, productivity, *, market: str = "domestic"):
        """Revenue over ``sigma`` less the fixed cost of serving the market."""
        revenue = self.revenue(productivity, market=market)
        fixed = (self.spec.domestic_fixed_cost if market == "domestic"
                 else self.spec.export_fixed_cost)
        return revenue / self.spec.sigma - fixed


def _checked(spec: MelitzSpec) -> None:
    positive = ("sigma", "pareto_shape", "productivity_min", "labour", "entry_cost",
                "domestic_fixed_cost", "export_fixed_cost", "iceberg_cost")
    for name in positive:
        value = getattr(spec, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if spec.sigma <= 1:
        raise ValueError(f"sigma must be above one, got {spec.sigma}")
    if spec.pareto_shape <= spec.sigma - 1:
        raise ValueError(f"pareto_shape must exceed sigma - 1 ({spec.sigma - 1}), got "
                         f"{spec.pareto_shape}")
    if spec.iceberg_cost < 1:
        raise ValueError(f"iceberg_cost is at least one, got {spec.iceberg_cost}")


def solve_melitz(spec: MelitzSpec | None = None) -> MelitzResult:
    """Solve the symmetric two-country model in closed form.

    Use ``dataclasses.replace`` on ``spec`` for a counterfactual. Exporters must be a positively
    selected subset of active firms: the equilibria in which every active firm exports or every
    entrant produces are refused rather than solved with the interior-cutoff equations, and so is
    autarky, since ``iceberg_cost`` must be finite.
    """
    spec = spec or MelitzSpec()
    _checked(spec)
    s = spec
    a = s.sigma - 1.0
    k = s.pareto_shape
    denominator = k - a
    cutoff_ratio = s.iceberg_cost * (s.export_fixed_cost / s.domestic_fixed_cost) ** (1.0 / a)
    if cutoff_ratio <= 1:
        raise ValueError("interior export selection requires iceberg_cost ** (sigma - 1) * "
                         "export_fixed_cost / domestic_fixed_cost > 1; the all-exporters corner "
                         "is not in this version")
    domestic_cutoff = s.productivity_min * (
        a / (s.entry_cost * denominator)
        * (s.domestic_fixed_cost + s.export_fixed_cost * cutoff_ratio ** -k)
    ) ** (1.0 / k)
    if domestic_cutoff < s.productivity_min:
        raise ValueError(f"the domestic cutoff {domestic_cutoff:.6g} is below productivity_min "
                         f"{s.productivity_min:.6g}; the all-entrants-produce corner is not in "
                         "this version")
    export_cutoff = cutoff_ratio * domestic_cutoff
    domestic_survival = (s.productivity_min / domestic_cutoff) ** k
    export_survival = (s.productivity_min / export_cutoff) ** k
    entrants = s.labour * a / (s.sigma * k * s.entry_cost)
    domestic_firms = entrants * domestic_survival
    exporters = entrants * export_survival
    rho = a / s.sigma
    price_power = entrants * rho ** a * k / denominator * (
        s.productivity_min ** k
        * (domestic_cutoff ** -denominator
           + s.iceberg_cost ** -a * export_cutoff ** -denominator)
    )
    price_index = price_power ** (-1.0 / a)
    welfare = s.labour / price_index
    average_revenue_factor = s.sigma * k / denominator
    domestic_revenue = domestic_firms * s.domestic_fixed_cost * average_revenue_factor
    export_revenue = exporters * s.export_fixed_cost * average_revenue_factor
    result = MelitzResult(
        spec=s,
        domestic_cutoff=float(domestic_cutoff),
        export_cutoff=float(export_cutoff),
        entrants=float(entrants),
        domestic_firms=float(domestic_firms),
        exporters=float(exporters),
        price_index=float(price_index),
        welfare=float(welfare),
        domestic_revenue=float(domestic_revenue),
        export_revenue=float(export_revenue),
        average_domestic_productivity=float((k / denominator) ** (1.0 / a) * domestic_cutoff),
        average_export_productivity=float((k / denominator) ** (1.0 / a) * export_cutoff),
        residual=0.0,
        residuals={},
    )
    fixed_labour = domestic_firms * s.domestic_fixed_cost + exporters * s.export_fixed_cost
    variable_labour = a / s.sigma * (domestic_revenue + export_revenue)
    residuals = {
        "domestic_zero_profit": float(result.operating_profit(domestic_cutoff)),
        "export_zero_profit": float(result.operating_profit(export_cutoff, market="export")),
        "free_entry": float(
            a / denominator * (s.domestic_fixed_cost * domestic_survival
                               + s.export_fixed_cost * export_survival) - s.entry_cost
        ),
        "labour": float(entrants * s.entry_cost + fixed_labour + variable_labour - s.labour),
        "revenue": float(domestic_revenue + export_revenue - s.labour),
    }
    return replace(result, residuals=residuals, residual=max(map(abs, residuals.values())))
