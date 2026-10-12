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

    #: Account that collects the import tariff from commodities (Hosoe's TRF).
    #: Absent, there is no tariff unless ``calibrate(..., tariffs=)`` supplies rates.
    #: The receipts are not product tax.
    tariff_account: str | None = None

    #: Account that collects the indirect tax on domestic sales from commodities
    #: (Hosoe's IDT, ``tauz``). Absent, that instrument is zero. It is not the
    #: product tax on absorption, and it is not the specific activity tax.
    #: Exports sold at the producer price are outside the wedge.
    indirect_tax_account: str | None = None
    activities: Sequence[str] | None = None
    commodities: Sequence[str] | None = None
    households: Sequence[str] | None = None

    # -- functional forms ------------------------------------------------
    value_added: str = "ces"                 # "ces" | "cobb_douglas" | "leontief"
    sigma_va: float | Mapping[str, float] = 0.5
    sigma_armington: float | Mapping[str, float] = 2.0
    export_rule: str = "fixed_shares"        # "fixed_shares" | "cet"
    sigma_cet: float | Mapping[str, float] = 2.0
    #: Export-demand elasticity. None: the economy is a price taker abroad and
    #: ``Scenario.export_price`` is what its exports fetch. A positive number η
    #: puts each exported commodity on a demand curve,
    #: ``E / E0 = (pWe / export_price)^(-η)``, the form of Hosoe's ``lrgcge``.
    #: At the base year ``E = E0`` and the world price is the calibrated one.
    #: A scalar, or a mapping by commodity in which a missing commodity (and no
    #: ``"__default__"``) stays a price taker. Read only under ``export_rule="cet"``.
    export_demand: float | Mapping[str, float] | None = None
    #: Import-supply elasticity, the mirror of ``export_demand``: None keeps the
    #: world import price exogenous (``Scenario.world_price``); a positive ψ
    #: makes it rise with the volume bought, ``M / M0 = (pWm / world_price)^ψ``.
    import_supply: float | Mapping[str, float] | None = None
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
    #: identity the solver was told to satisfy. With ``regions`` set, name the
    #: region too: ``"r1:f_lab"``, not ``"f_lab"``.
    numeraire_factor: str | None = None

    #: Regions, in order. Absent, the model is the one-region model and account
    #: names are not split. Present, every activity, commodity, household and
    #: factor is ``"{region}:{local}"`` and a CES across regions of origin sits
    #: under the Armington nest. One entry is a labelled one-region model.
    regions: Sequence[str] | None = None

    #: Elasticity of substitution across regions of origin, inside the national
    #: domestic composite. A scalar, or a mapping from the local commodity name.
    #: Zero is Leontief: the base-year origin mix does not answer to prices.
    sigma_interregional: float | Mapping[str, float] = 2.0

    def __post_init__(self):
        self.factors = list(self.factors)
        if self.regions is not None:
            self.regions = tuple(self.regions)
            if len(self.regions) < 1:
                raise ValueError("regions is empty")
            if len(set(self.regions)) != len(self.regions):
                raise ValueError(f"region names repeat: {self.regions}")
        if not self.regions:
            if self.numeraire_factor is None:
                self.numeraire_factor = self.factors[0]
            if self.numeraire_factor not in self.factors:
                raise ValueError(
                    f"numeraire {self.numeraire_factor!r} is not among {self.factors}")
        elif self.numeraire_factor is None:
            raise ValueError(
                "a regional model needs numeraire_factor as 'region:factor', "
                "for example 'r1:f_lab'. One national numeraire, one unsolved market.")
        if not isinstance(self.sigma_interregional, Mapping):
            if float(self.sigma_interregional) < 0:
                raise ValueError(
                    "sigma_interregional must be >= 0, "
                    f"got {self.sigma_interregional}")
        for field_name, allowed in (
            ("value_added", ("ces", "cobb_douglas", "leontief")),
            ("export_rule", ("fixed_shares", "cet")),
            ("household_demand", ("cobb_douglas", "les")),
            ("negative_value_added", ("leontief", "net")),
        ):
            got = getattr(self, field_name)
            if got not in allowed:
                raise ValueError(f"{field_name} must be one of {allowed}, got {got!r}")
        for field_name in ("export_demand", "import_supply"):
            got = getattr(self, field_name)
            if got is None:
                continue
            values = got.values() if isinstance(got, Mapping) else [got]
            if any(not float(v) > 0 for v in values):
                raise ValueError(
                    f"{field_name} must be a positive elasticity, got {got!r}. "
                    f"Omit it to keep the economy a price taker.")
        if self.export_demand is not None and self.export_rule != "cet":
            raise ValueError(
                "export_demand needs export_rule='cet'. With fixed export shares the "
                "export volume is a fixed fraction of output, so no world price could "
                "clear a demand curve for it.")

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

    def prepare(self, sam) -> "ModelSpec":
        """Expand regional account lists. A model without ``regions`` is returned as-is.

        Local factor names (``f_lab``) become ``r1:f_lab`` for every region when
        the short name is not itself an account. Explicit activity, commodity and
        household lists are kept, and checked. Absent, they are discovered: the
        region is before the colon and the local name still obeys the prefix.
        """
        if not self.regions:
            return self
        import dataclasses
        regions = tuple(self.regions)
        acc = list(sam.accounts)
        acc_set = set(acc)

        def split(name):
            reg, sep, local = name.partition(":")
            if not sep or reg not in regions:
                raise ValueError(
                    f"account {name!r} is in a regional model but is not "
                    f"'region:name' with region one of {list(regions)}")
            return reg, local

        def discover(prefix):
            found = []
            for name in acc:
                if ":" not in name:
                    continue
                reg, local = name.split(":", 1)
                if reg in regions and local.startswith(prefix):
                    found.append(name)
            found.sort(key=lambda n: (regions.index(n.split(":", 1)[0]), n.split(":", 1)[1]))
            if not found:
                raise ValueError(
                    f"no regional accounts with local prefix {prefix!r} in regions {list(regions)}")
            return found

        def take(explicit, prefix, label):
            if explicit is None:
                return discover(prefix)
            names = list(explicit)
            missing = [n for n in names if n not in acc_set]
            if missing:
                raise KeyError(f"{label} not present in the SAM: {missing}")
            for n in names:
                split(n)
            return names

        acts = take(self.activities, self.activity_prefix, "activities")
        coms = take(self.commodities, self.commodity_prefix, "commodities")
        hhs = take(self.households, self.household_prefix, "households")

        if all(f in acc_set for f in self.factors):
            factors = list(self.factors)
            for f in factors:
                split(f)
        else:
            factors = [f"{reg}:{f}" for reg in regions for f in self.factors]
            missing = [f for f in factors if f not in acc_set]
            if missing:
                raise KeyError(
                    f"regional factors not in the SAM: {missing}. Pass full names "
                    f"('r1:f_lab') or local names that exist in every region.")
        if self.numeraire_factor not in factors:
            raise ValueError(
                f"numeraire {self.numeraire_factor!r} is not among the regional "
                f"factors {factors}")
        return dataclasses.replace(
            self, activities=acts, commodities=coms, households=hhs, factors=factors)

    def sigma_va_vector(self, activities: Sequence[str]):
        return _as_vector(self.sigma_va, activities, "sigma_va")

    def sigma_armington_vector(self, commodities: Sequence[str]):
        return _as_vector(self.sigma_armington, commodities, "sigma_armington")

    def sigma_cet_vector(self, commodities: Sequence[str]):
        import numpy as np
        v = _as_vector(self.sigma_cet, commodities, "sigma_cet")
        if np.any(~np.isfinite(v)) or np.any(v < 0):
            raise ValueError("sigma_cet must be finite and non-negative for every commodity")
        return v

    def export_demand_vector(self, commodities: Sequence[str]):
        """η by commodity; ``inf`` marks a price taker."""
        return _elasticity_or_inf(self.export_demand, commodities)

    def import_supply_vector(self, commodities: Sequence[str]):
        """ψ by commodity; ``inf`` marks a price taker."""
        return _elasticity_or_inf(self.import_supply, commodities)


def _elasticity_or_inf(value, names):
    import numpy as np
    if value is None:
        return np.full(len(names), np.inf)
    if isinstance(value, Mapping):
        default = value.get("__default__", np.inf)
        return np.array([float(value.get(n, default)) for n in names])
    return np.full(len(names), float(value))


def _as_vector(value, names, label):
    import numpy as np
    if isinstance(value, Mapping):
        return np.array([float(value.get(n, value.get("__default__", np.nan))) for n in names])
    return np.full(len(names), float(value))
