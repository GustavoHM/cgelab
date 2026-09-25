"""Model specification: which accounts are what, and which functional forms apply.

Kept apart from the calibration on purpose. The SAM says what the base year was;
the specification says what the modeller assumes about substitution, and those
are different kinds of statement with different evidence behind them. Every
elasticity here should be traceable to a source or declared as a guess and swept
in robustness -- the package cannot enforce that, but it can at least keep the
assumptions in one visible object instead of scattering them through the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = ["ModelSpec", "AccountLayout"]


@dataclass
class AccountLayout:
    """The ordered account lists a calibrated economy is built on."""

    activities: list
    commodities: list
    households: list
    factors: list
    government: str
    investment: str
    rest_of_world: str

    def __repr__(self) -> str:
        return (f"AccountLayout({len(self.activities)} activities, "
                f"{len(self.commodities)} commodities, {len(self.households)} households, "
                f"{len(self.factors)} factors)")


@dataclass
class ModelSpec:
    """Account classification and functional forms.

    Accounts are found by prefix unless an explicit list is given. Explicit lists
    win, and are the safer choice for a SAM whose naming is not systematic: a
    prefix rule that silently misses an account produces a model that balances
    and is wrong.
    """

    # -- classification --------------------------------------------------
    activity_prefix: str = "a_"
    commodity_prefix: str = "c_"
    household_prefix: str = "hh_"
    factors: Sequence[str] = ("f_lab", "f_cap")
    government: str = "gov"
    investment: str = "s_i"
    rest_of_world: str = "row"
    activities: Sequence[str] | None = None
    commodities: Sequence[str] | None = None
    households: Sequence[str] | None = None

    # -- functional forms ------------------------------------------------
    value_added: str = "ces"                 # "ces" | "cobb_douglas" | "leontief"
    sigma_va: float | Mapping[str, float] = 0.5
    sigma_armington: float | Mapping[str, float] = 2.0
    export_rule: str = "fixed_shares"        # "fixed_shares" | "cet"
    sigma_cet: float = 2.0
    household_demand: str = "cobb_douglas"   # "cobb_douglas" | "les"
    frisch: float = -1.5                     # LES only: Frisch parameter, sets subsistence
    margins: bool = True

    #: What to do with an activity whose value added is negative in the base
    #: year. It happens -- an operating surplus can be negative in a year a
    #: sector collapsed, and that is real data, not an error to be cleaned.
    #: No CES nest accepts a negative exponent, so either the activity is put on
    #: fixed proportions ("leontief", the default, which keeps the SAM's factor
    #: quantities intact) or its factor demands are rescaled to the net total
    #: ("net", which changes them and so requires aggregate factor supply to
    #: fall by the same amount, or the base year will not reproduce).
    negative_value_added: str = "leontief"   # "leontief" | "net"

    #: Numeraire factor. Its price is fixed and its market is left to Walras'
    #: law, so the residual on it is an INDEPENDENT check rather than an
    #: identity the solver was told to satisfy.
    numeraire_factor: str | None = None

    def __post_init__(self):
        self.factors = list(self.factors)
        if self.numeraire_factor is None:
            self.numeraire_factor = self.factors[0]
        if self.numeraire_factor not in self.factors:
            raise ValueError(f"numeraire {self.numeraire_factor!r} is not among {self.factors}")
        for field_name, allowed in (
            ("value_added", ("ces", "cobb_douglas", "leontief")),
            ("export_rule", ("fixed_shares", "cet")),
            ("household_demand", ("cobb_douglas", "les")),
            ("negative_value_added", ("leontief", "net")),
        ):
            got = getattr(self, field_name)
            if got not in allowed:
                raise ValueError(f"{field_name} must be one of {allowed}, got {got!r}")

    # -- helpers ---------------------------------------------------------
    def classify(self, sam) -> AccountLayout:
        """Split the SAM's accounts into the model's blocks."""
        acc = sam.accounts

        def pick(explicit, prefix, label):
            if explicit is not None:
                unknown = [a for a in explicit if a not in acc]
                if unknown:
                    raise KeyError(f"{label} not present in the SAM: {unknown}")
                return list(explicit)
            found = sorted(a for a in acc if a.startswith(prefix))
            if not found:
                raise ValueError(
                    f"no {label} found with prefix {prefix!r}; pass an explicit list")
            return found

        acts = pick(self.activities, self.activity_prefix, "activities")
        coms = pick(self.commodities, self.commodity_prefix, "commodities")
        hhs = pick(self.households, self.household_prefix, "households")
        missing = [f for f in self.factors if f not in acc]
        if missing:
            raise KeyError(f"factors not present in the SAM: {missing}")
        for role in ("government", "investment", "rest_of_world"):
            nm = getattr(self, role)
            if nm not in acc:
                raise KeyError(f"the {role} account {nm!r} is not in the SAM")
        return AccountLayout(activities=acts, commodities=coms, households=hhs,
                             factors=list(self.factors), government=self.government,
                             investment=self.investment, rest_of_world=self.rest_of_world)

    def sigma_va_vector(self, activities: Sequence[str]):
        return _as_vector(self.sigma_va, activities, "sigma_va")

    def sigma_armington_vector(self, commodities: Sequence[str]):
        return _as_vector(self.sigma_armington, commodities, "sigma_armington")


def _as_vector(value, names, label):
    import numpy as np
    if isinstance(value, Mapping):
        return np.array([float(value.get(n, value.get("__default__", np.nan))) for n in names])
    return np.full(len(names), float(value))
