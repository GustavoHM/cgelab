"""Macroeconomic closure: which balances are imposed and which agent adjusts.

A CGE model is not determined by its equations alone. Investment can be driven by
saving or fixed in real terms; the current account can be pinned with the real
exchange rate clearing it, or the exchange rate fixed with foreign saving
adjusting; government demand can be fixed in real or in nominal terms. Each
choice is a different model of who bears an adjustment, and results can turn on
it, so the choice belongs in a named object that travels with the results rather
than in a branch inside the solver.

One rule is not a switch, because it is what makes the solution checkable: the
NUMERAIRE factor's market is never solved. It is left to Walras' law, so its
residual is an independent test of the accounting rather than an identity the
solver was instructed to satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .segments import LabourMarket

__all__ = ["Closure", "LabourMarket"]


@dataclass
class Closure:
    """Closure rules for one period."""

    #: Level of the numeraire factor price. Only for the homogeneity test: no
    #: real magnitude may depend on it, and every exogenous nominal magnitude
    #: denominated in the numeraire co-scales with it -- foreign saving, the
    #: trade-balance target and the specific activity tax, which enters the
    #: activity price additively rather than ad valorem. Leaving any of them
    #: fixed while the price level moves would shift the real exchange rate, real
    #: investment, or the real tax wedge, and the test would fail for a reason
    #: that has nothing to do with the model.
    numeraire: float = 1.0

    #: ``"savings_driven"`` -- investment absorbs whatever household, government
    #: and foreign saving amount to. ``"fixed_real"`` -- investment quantities
    #: are held and household saving adjusts residually.
    investment: str = "savings_driven"

    #: ``"real_fixed"`` -- government commodity demand is held in QUANTITIES, so
    #: its nominal spending moves with prices. ``"nominal_fixed"`` -- its budget
    #: is held and its real purchases move instead. ``"saving_rate"`` -- it
    #: saves the base-year share of its income (taxes, factor income, net of
    #: transfers to households) and spends the rest in base-year value shares,
    #: the rule of Hosoe's ``stdcge``. A tax cut then cuts public consumption.
    government: str = "real_fixed"

    #: ``"fixed_saving"`` -- the current account is pinned at its base-year value
    #: and the exchange rate clears it. ``"fixed_rate"`` -- the exchange rate is
    #: exogenous and foreign saving adjusts.
    foreign: str = "fixed_saving"

    #: Unit in which ``foreign="fixed_saving"`` holds foreign saving.
    #: ``"numeraire"`` -- fixed in the numeraire's units. ``"foreign_currency"``
    #: -- fixed in dollars, as in Hosoe's ``stdcge`` and the IFPRI standard
    #: model, so a depreciation raises what it finances at home.
    foreign_saving_unit: str = "numeraire"

    #: A target for the government account. ``None`` -- government saving is
    #: whatever revenue and spending leave (the default). ``"saving"`` -- it is
    #: held at ``government_target_value`` (the base-year saving when None): a
    #: deficit target. ``"revenue"`` -- total revenue is held there instead, so a
    #: scenario that cuts one tax has another make up the receipts:
    #: revenue-neutral tax replacement. Both are in numeraire units.
    government_target: str | None = None
    government_target_value: float | None = None
    #: What moves to meet the target: ``"direct_tax"`` (every household's
    #: direct tax rate, scaled by one common factor), ``"product_tax"`` (every
    #: product tax rate, consumer and firm, scaled likewise) or ``"spending"``
    #: (government consumption, in base-year value shares; saving target only).
    government_adjusts: str = "direct_tax"

    #: Exogenous exchange rate, used only under ``foreign="fixed_rate"``.
    exchange_rate: float = 1.0

    #: Tolerance on the solved market gaps. Convergence is judged on the
    #: RESIDUAL and never on the solver's success flag: `hybr` reports failure
    #: whenever its finite-difference Jacobian stops improving, which includes
    #: the case where the starting point already IS the solution.
    tolerance: float = 1e-9

    #: ``"full_employment"`` -- every factor market clears and a shock shows up
    #: in the wage. ``"fixed_real_wage"`` -- the real wage is pinned and
    #: employment adjusts (Keynesian). ``"wage_curve"`` -- the real wage falls
    #: as unemployment rises above ``natural_unemployment``. ``"elastic_supply"``
    #: -- labour supply rises with the real wage. ``"wage_floor"`` -- the real
    #: wage cannot fall below ``real_wage``: real wage − floor ≥ 0 ⊥
    #: unemployment ≥ 0, full employment while the floor is slack. The last
    #: four refuse to use the labour factor as numeraire: that market is the
    #: Walras check, and a closure that also pins its wage turns the check into
    #: an identity.
    labour: str = "full_employment"
    labour_factor: str = "f_lab"
    #: Real wage under ``fixed_real_wage``, and the floor under ``wage_floor``,
    #: in base-year units (1 is the calibrated wage). Absent, the calibrated
    #: real wage.
    real_wage: float | None = None
    #: Blanchflower–Oswald β > 0. Read only under ``wage_curve``.
    wage_curve_elasticity: float = 0.1
    #: Unemployment rate at which the wage curve pays the calibrated real wage.
    #: The SAM records employment, not the labour force, so the force is
    #: ``employment / (1 - u0)`` and the base year sits AT this rate. Zero is
    #: refused: the log curve is undefined there.
    natural_unemployment: float = 0.0
    #: ε ≥ 0 of ``L = L0 * (real_wage / real_wage0) ** ε``. Zero is inelastic.
    labour_supply_elasticity: float = 0.0

    #: How each group of ``ModelSpec.labour_segments`` clears,
    #: ``{group: LabourMarket}``. A group left out has no rigid segment: its
    #: segments share one wage, and its last segment absorbs the rest of each
    #: household's workforce.
    labour_markets: Mapping[str, LabourMarket] = field(default_factory=dict)

    def __post_init__(self):
        for name, allowed in (("investment", ("savings_driven", "fixed_real")),
                              ("government", ("real_fixed", "nominal_fixed", "saving_rate")),
                              ("foreign", ("fixed_saving", "fixed_rate")),
                              ("foreign_saving_unit", ("numeraire", "foreign_currency")),
                              ("labour", ("full_employment", "fixed_real_wage",
                                          "wage_curve", "elastic_supply", "wage_floor"))):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"closure.{name} must be one of {allowed}, got {got!r}")
        if self.government_target not in (None, "saving", "revenue"):
            raise ValueError("closure.government_target must be None, 'saving' or 'revenue', "
                             f"got {self.government_target!r}")
        if self.government_adjusts not in ("direct_tax", "product_tax", "spending"):
            raise ValueError("closure.government_adjusts must be 'direct_tax', 'product_tax' "
                             f"or 'spending', got {self.government_adjusts!r}")
        if self.government_target is not None and self.government_adjusts == "spending":
            if self.government_target == "revenue":
                raise ValueError(
                    "a revenue target needs a tax to move: spending does not change revenue. "
                    "Use government_adjusts='direct_tax' or 'product_tax'.")
            if self.government == "saving_rate":
                raise ValueError(
                    "government='saving_rate' already sets spending from income; a saving "
                    "target met by spending would be a second rule for the same quantity.")
        if self.labour == "wage_curve":
            if not 0.0 < self.natural_unemployment < 1.0:
                raise ValueError(
                    "closure.natural_unemployment must lie strictly between 0 and 1 "
                    f"under wage_curve, got {self.natural_unemployment}. The SAM records "
                    "employment, and the base year sits at this rate; zero unemployment "
                    "makes the log wage curve undefined.")
            if self.wage_curve_elasticity <= 0:
                raise ValueError(
                    "closure.wage_curve_elasticity must be positive, "
                    f"got {self.wage_curve_elasticity}")
        if self.labour == "elastic_supply" and self.labour_supply_elasticity < 0:
            raise ValueError(
                "closure.labour_supply_elasticity must be >= 0, "
                f"got {self.labour_supply_elasticity}")
        for name, mkt in dict(self.labour_markets or {}).items():
            if not isinstance(mkt, LabourMarket):
                raise TypeError(f"closure.labour_markets[{name!r}] must be a LabourMarket, "
                                f"got {type(mkt).__name__}")
        if self.real_wage is not None and self.real_wage <= 0:
            raise ValueError(f"closure.real_wage must be positive, got {self.real_wage}")

    @property
    def solves_exchange_rate(self) -> bool:
        return self.foreign == "fixed_saving"

    def describe(self) -> str:
        text = (f"numeraire={self.numeraire:g}; investment={self.investment}; "
                f"government={self.government}; foreign={self.foreign}")
        if self.government_target is not None:
            text += f"; government {self.government_target} target, {self.government_adjusts} adjusts"
        if self.foreign_saving_unit != "numeraire":
            text += f" ({self.foreign_saving_unit})"
        if self.labour != "full_employment":
            text += f"; labour={self.labour}({self.labour_factor})"
        for name, mkt in dict(self.labour_markets or {}).items():
            kind = "harris_todaro" if mkt.unemployment == "harris_todaro" else "dual"
            text += f"; {name}: {kind}, rigid {sorted(mkt.rigid)}"
        return text
