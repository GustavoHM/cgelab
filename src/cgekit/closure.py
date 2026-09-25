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

from dataclasses import dataclass

__all__ = ["Closure"]


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
    #: is held and its real purchases move instead.
    government: str = "real_fixed"

    #: ``"fixed_saving"`` -- the current account is pinned at its base-year value
    #: and the exchange rate clears it. ``"fixed_rate"`` -- the exchange rate is
    #: exogenous and foreign saving adjusts.
    foreign: str = "fixed_saving"

    #: Exogenous exchange rate, used only under ``foreign="fixed_rate"``.
    exchange_rate: float = 1.0

    #: Tolerance on the solved market gaps. Convergence is judged on the
    #: RESIDUAL and never on the solver's success flag: `hybr` reports failure
    #: whenever its finite-difference Jacobian stops improving, which includes
    #: the case where the starting point already IS the solution.
    tolerance: float = 1e-9

    def __post_init__(self):
        for name, allowed in (("investment", ("savings_driven", "fixed_real")),
                              ("government", ("real_fixed", "nominal_fixed")),
                              ("foreign", ("fixed_saving", "fixed_rate"))):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"closure.{name} must be one of {allowed}, got {got!r}")

    @property
    def solves_exchange_rate(self) -> bool:
        return self.foreign == "fixed_saving"

    def describe(self) -> str:
        return (f"numeraire={self.numeraire:g}; investment={self.investment}; "
                f"government={self.government}; foreign={self.foreign}")
