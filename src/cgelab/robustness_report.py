"""One-call robustness reports assembled from the public analysis tools."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .robustness import closure_grid, closure_sweep, in_money
from .sam_uncertainty import sam_draws, sam_sweep
from .sensitivity import (Normal, Triangular, Uniform, _as_dict, _replace_spec, sensitivity,
                          sobol as sobol_analysis, spec_sweep)

__all__ = ["RobustnessReport", "robustness_report"]


_ROSE = "Rose et al. (2023), endpoints; triangular shape is a package convention"
_ARNDT = "Arndt, Robinson and Tarp (2002), prior support"
_RELATIVE = {"sigma_va", "sigma_armington", "sigma_cet"}


@dataclass
class RobustnessReport:
    """Summary tables, figures and the full results that produced them."""

    tables: dict[str, pd.DataFrame]
    parts: dict = field(repr=False)
    figures: dict = field(default_factory=dict, repr=False)

    @property
    def ranges(self):
        return self.tables["ranges"]

    def __repr__(self):
        return f"RobustnessReport({len(self.tables)} tables, {len(self.figures)} figures)"


def _active_centres(spec):
    fields = ["sigma_va", "sigma_armington"]
    if spec.export_rule == "cet":
        fields.append("sigma_cet")
    if spec.energy is not None:
        fields += ["sigma_vae", "sigma_energy"]
        if spec.energy_nest == "capital":
            fields.append("sigma_ke")
    if spec.regions is not None:
        fields.append("sigma_interregional")
    if spec.vintages is not None:
        fields += ["vintage_sigma", "vintage_substitution"]
    for name in ("factor_mobility", "segment_mobility", "export_demand", "import_supply"):
        if getattr(spec, name) is not None:
            fields.append(name)
    centres = {}
    for name in fields:
        value = getattr(spec, name)
        if isinstance(value, Mapping):
            for account, centre in value.items():
                if account != "__default__":
                    centres[f"{name}[{account}]"] = float(centre)
        else:
            centres[name] = float(value)
    return centres


def _distributions(spec, ranges, parameters):
    centres = _active_centres(spec)
    custom_ranges = dict(ranges) if isinstance(ranges, Mapping) else {}
    custom_parameters = dict(parameters or {})
    valid_fields = {f.name for f in fields(spec)}
    for name in list(custom_ranges) + list(custom_parameters):
        if name.split("[")[0] not in valid_fields:
            raise KeyError(f"{name.split('[')[0]!r} is not a ModelSpec field")
        centres.setdefault(name, np.nan)
    preset = ranges if isinstance(ranges, str) else "rose2023"
    if preset not in ("rose2023", "arndt2002"):
        raise ValueError("ranges must be 'rose2023', 'arndt2002', or a mapping")
    distributions, sources = {}, {}
    missing = []
    for name, centre in centres.items():
        field_name = name.split("[")[0]
        if name in custom_parameters:
            distributions[name] = custom_parameters[name]
            sources[name] = "user distribution"
        elif name in custom_ranges:
            low, mode, high = map(float, custom_ranges[name])
            distributions[name] = Triangular(low, mode, high)
            sources[name] = "user range; triangular shape is a package convention"
        elif preset == "arndt2002" and field_name in ("sigma_armington", "sigma_cet"):
            high = 9.0 if field_name == "sigma_armington" else 5.0
            distributions[name] = Triangular(0.3, 1.5, high)
            sources[name] = _ARNDT
        elif preset == "rose2023" and field_name in _RELATIVE and centre > 0:
            distributions[name] = Triangular(0.5 * centre, centre, 1.5 * centre)
            sources[name] = _ROSE
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"no sourced default range for {missing}; pass ranges= or parameters=")
    for name, distribution in distributions.items():
        if not hasattr(distribution, "draw") or not hasattr(distribution, "mean"):
            raise TypeError(f"parameter {name!r} is not a sensitivity distribution")
    return distributions, sources


def _range_table(distributions, sources):
    rows = {}
    for name, distribution in distributions.items():
        if isinstance(distribution, Triangular):
            low, mode, high = distribution.low, distribution.mode, distribution.high
        elif isinstance(distribution, Uniform):
            low, mode, high = distribution.low, distribution.mean, distribution.high
        elif isinstance(distribution, Normal):
            low, mode = distribution.mean - 2 * distribution.sd, distribution.mean
            high = distribution.mean + 2 * distribution.sd
        else:
            low, mode = distribution.mean - 2 * distribution.sd, distribution.mean
            high = distribution.mean + 2 * distribution.sd
        rows[name] = {"low": low, "mode": mode, "high": high, "source": sources[name]}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("parameter")


def _one_at_a_time(run, range_table, outputs):
    centres = range_table["mode"].to_dict()
    records, failed = [], []
    for parameter, row in range_table.iterrows():
        for point in ("low", "mode", "high"):
            values = dict(centres)
            values[parameter] = row[point]
            try:
                output = _as_dict(run(values))
            except (ValueError, RuntimeError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                failed.append({"parameter": parameter, "point": point,
                               "reason": f"{type(exc).__name__}: {exc}"})
                continue
            for name, value in output.items():
                records.append({"output": name, "parameter": parameter,
                                "point": point, "value": float(value)})
    long = pd.DataFrame(records)
    table = long.pivot(index=["output", "parameter"], columns="point", values="value")
    table = table.reindex(columns=["low", "mode", "high"])
    table = table.rename(columns={"mode": "central"})[["low", "central", "high"]]
    index = pd.MultiIndex.from_product([outputs, range_table.index],
                                       names=["output", "parameter"])
    table = table.reindex(index)
    table["min"] = table[["low", "central", "high"]].min(axis=1)
    table["max"] = table[["low", "central", "high"]].max(axis=1)
    table["spread"] = table["max"] - table["min"]
    table["argmin"] = table[["low", "central", "high"]].idxmin(axis=1)
    table["argmax"] = table[["low", "central", "high"]].idxmax(axis=1)
    low_change = table["low"] - table["central"]
    high_change = table["high"] - table["central"]
    table["same_sign"] = ((np.sign(low_change) == np.sign(high_change))
                          & (low_change != 0) & (high_change != 0))
    failures = pd.DataFrame(failed, columns=["parameter", "point", "reason"])
    return table, failures


def robustness_report(sam, spec, *, measure: Callable, scenario=None, sam_blocks,
                      ranges="rose2023", parameters: Mapping | None = None,
                      closures=None, draws: int = 200, seed: int | None = 20260724,
                      sobol: bool = False, sobol_n: int = 256,
                      money: bool = True, unit: float = 1.0, label: str | None = None,
                      figures: bool = False, calibrate_kwargs: Mapping | None = None,
                      sam_kwargs: Mapping | None = None) -> RobustnessReport:
    """Run the robustness tools together and return their tables and full results.

    The default ``rose2023`` ranges are ``0.5x, x, 1.5x`` for value-added,
    Armington and export-CET elasticities, with a triangular shape chosen by the
    package. ``arndt2002`` supplies the published trade-elasticity supports;
    other active elasticity fields need explicit ``ranges`` or ``parameters``.
    ``sam_blocks`` is required and passed to :func:`sam_draws`; ``sam_kwargs``
    controls that draw. Sobol indices and matplotlib figures are opt-in.
    """
    distributions, sources = _distributions(spec, ranges, parameters)
    range_table = _range_table(distributions, sources)
    run = spec_sweep(sam, spec, measure=measure, scenario=scenario,
                     calibrate_kwargs=calibrate_kwargs)
    if all(getattr(d, "symmetric", False) for d in distributions.values()):
        quadrature = sensitivity(run, distributions, seed=seed)
        quadrature_table = quadrature.summary()
    else:
        quadrature = None
        quadrature_table = pd.DataFrame()
        quadrature_table.attrs["skipped"] = (
            "Stroud's rule needs symmetric distributions; use Monte Carlo or Sobol")
    monte_carlo = sensitivity(run, distributions, method="monte_carlo", draws=draws, seed=seed)
    outputs = monte_carlo.outputs
    oat, oat_failed = _one_at_a_time(run, range_table, outputs)

    central_spec = _replace_spec(spec, range_table["mode"].to_dict())
    from .calibrate import calibrate
    from .solve import solve_period
    central_econ = calibrate(sam, central_spec, **dict(calibrate_kwargs or {}))
    closure_specs = (closure_grid(investment=["savings_driven", "fixed_real"],
                                  foreign=["fixed_saving", "fixed_rate"])
                     if closures is None else closures)
    closure_results = closure_sweep(central_econ, closure_specs, scenario=scenario, measure=measure)
    drawn_sams = sam_draws(sam, blocks=sam_blocks, draws=draws, seed=seed,
                           **dict(sam_kwargs or {}))
    sam_results = sam_sweep(drawn_sams, spec, measure=measure, scenario=scenario,
                            parameters=distributions, calibrate_kwargs=calibrate_kwargs)
    tables = {"ranges": range_table, "oat": oat, "oat_failed": oat_failed,
              "quadrature": quadrature_table, "monte_carlo": monte_carlo.summary(),
              "closures": closure_results.table, "sam": sam_results.summary()}
    parts = {"quadrature": quadrature, "monte_carlo": monte_carlo,
             "closures": closure_results, "sam": sam_results}
    if money:
        base = solve_period(central_econ)
        alternative = solve_period(central_econ, scenario=scenario, warm_start=base)
        tables["money"] = in_money(base, alternative, unit=unit, label=label)
    if sobol:
        sobol_results = sobol_analysis(run, distributions, n=sobol_n, seed=seed)
        tables["sobol"] = sobol_results.indices()
        parts["sobol"] = sobol_results
    report = RobustnessReport(tables=tables, parts=parts)
    if figures:
        report.figures = _make_figures(report)
    return report


def _make_figures(report):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("figures=True needs matplotlib; install cgelab[plot]") from exc
    figures = {}
    for output in report.parts["monte_carlo"].outputs:
        table = report.tables["oat"].loc[output]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(table) + 1)))
        centre = table["central"].to_numpy()
        low = table["low"].to_numpy() - centre
        high = table["high"].to_numpy() - centre
        y = np.arange(len(table))
        ax.barh(y, high - low, left=low, color="#4c78a8")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, table.index)
        ax.set_title(f"One-at-a-time sensitivity: {output}")
        ax.set_xlabel("Change from the central result")
        fig.tight_layout()
        figures[f"tornado_{output}"] = fig

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(report.parts["monte_carlo"].points[output], bins=20,
                alpha=0.55, label="elasticities")
        ax.hist(report.parts["sam"].points[output], bins=20,
                alpha=0.55, label="SAM and elasticities")
        ax.set_title(f"Robustness distribution: {output}")
        ax.legend()
        fig.tight_layout()
        figures[f"distribution_{output}"] = fig
        if "sobol" in report.parts:
            indices = report.tables["sobol"].loc[output]
            x = np.arange(len(indices))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - 0.2, indices["S1"], width=0.4, label="S1")
            ax.bar(x + 0.2, indices["ST"], width=0.4, label="ST")
            ax.set_xticks(x, indices.index, rotation=45, ha="right")
            ax.set_title(f"Sobol indices: {output}")
            ax.legend()
            fig.tight_layout()
            figures[f"sobol_{output}"] = fig
    return figures
