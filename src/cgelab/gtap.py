"""Aggregation of GTAP headers and construction of a one-region SAM."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .build import SupplyUse, sut_to_sam
from .har import Header

__all__ = ["aggregate_gtap", "gtap_supply_use", "gtap_sam"]


def _groups(labels, rule):
    if not isinstance(rule, Mapping):
        raise TypeError("each aggregation rule must map old labels to a new label")
    out = []
    for label in labels:
        if label not in rule:
            raise KeyError(f"aggregation has no target for {label!r}")
        target = rule[label]
        if target not in out:
            out.append(target)
    index = np.array([out.index(rule[label]) for label in labels], dtype=int)
    return out, index


def _sum_groups(array, index, n, axis):
    return np.stack([array.take(np.where(index == k)[0], axis=axis).sum(axis=axis)
                     for k in range(n)], axis=axis)


def aggregate_gtap(headers, groups, *, values, weighted=None) -> dict:
    """Aggregate declared value headers, with explicit weights for parameters.

    ``groups`` maps axis positions to ``{old label: new label}``. Only names in
    ``values`` are summed. ``weighted`` maps another header to ``(weights,
    reduce_axes)``; weights are a Header or an array, reduced by position before
    the weighted mean. This keeps elasticities out of the summation path.
    """
    weighted = weighted or {}
    requested = list(values) + [name for name in weighted if name not in values]
    out = {}
    for name in requested:
        if name not in headers:
            raise KeyError(f"GTAP header {name!r} is missing")
        header = headers[name]
        array = np.asarray(header.array, dtype=float)
        axes = list(groups)
        if any(axis >= array.ndim or axis < 0 for axis in axes):
            raise ValueError(f"{name}: aggregation axis outside {array.ndim} dimensions")
        labels = list(header.labels)
        if len(labels) != array.ndim:
            raise ValueError(f"{name}: labels do not match array dimensions")
        if name in weighted:
            weight, reduce_axes = weighted[name]
            weight = np.asarray(weight.array if isinstance(weight, Header) else weight,
                                dtype=float)
            reduce_axes = tuple(sorted(reduce_axes, reverse=True))
            if any(axis >= weight.ndim or axis < 0 for axis in reduce_axes):
                raise ValueError(f"{name}: weight reduction axis is invalid")
            numerator = array * weight
            if reduce_axes:
                numerator = numerator.sum(axis=reduce_axes)
                weight = weight.sum(axis=reduce_axes)
            if numerator.shape != weight.shape:
                raise ValueError(f"{name}: weighted value and weight shapes differ")
            array, denominator = numerator, weight
        else:
            denominator = None
        new_labels = list(labels)
        for axis in axes:
            rule = groups[axis]
            targets, index = _groups(labels[axis], rule)
            array = _sum_groups(array, index, len(targets), axis)
            if denominator is not None:
                denominator = _sum_groups(denominator, index, len(targets), axis)
            new_labels[axis] = targets
        if denominator is not None:
            array = np.divide(array, denominator, out=np.zeros_like(array),
                              where=np.abs(denominator) > 1e-300)
        out[name] = Header(name, f"aggregated: {header.description}", header.kind,
                           array, list(header.sets), new_labels)
    return out


def _array(headers, name):
    if name not in headers:
        raise KeyError(f"GTAP header {name!r} is required for a one-region SAM")
    return np.asarray(headers[name].array, dtype=float)


def _labels(headers, name, axis):
    return list(headers[name].labels[axis])


def _region(headers, name, region):
    labels = _labels(headers, name, -1)
    if isinstance(region, str):
        if region not in labels:
            raise KeyError(f"region {region!r} is not in {labels}")
        return labels.index(region)
    index = int(region)
    if index < 0 or index >= len(labels):
        raise IndexError(f"region index {index} outside {len(labels)} regions")
    return index


def _take(headers, name, axis, index):
    return np.take(_array(headers, name), index, axis=axis)


def gtap_supply_use(headers, region=0) -> SupplyUse:
    """Map one GTAP region to supply-use tables, preserving FOB/CIF and taxes."""
    r = _region(headers, "TVOM", region)
    activities = _labels(headers, "VDFM", 1)
    commodities = _labels(headers, "VDFM", 0)
    if len(activities) != len(commodities):
        raise ValueError("a one-region SAM currently needs one activity per commodity")
    activities = [f"a_{commodity}" for commodity in commodities]
    output = _take(headers, "TVOM", -1, r)
    if output.shape != (len(activities),):
        raise ValueError("TVOM must have activities by region")
    make = pd.DataFrame(np.diag(output), index=activities, columns=commodities)
    use = (_take(headers, "VDFA", -1, r) + _take(headers, "VIFA", -1, r))
    use = pd.DataFrame(use, index=commodities, columns=activities)
    value_added = pd.DataFrame(_take(headers, "EVFA", -1, r),
                               index=_labels(headers, "EVFA", 0), columns=activities)

    private_market = _take(headers, "VDPM", -1, r) + _take(headers, "VIPM", -1, r)
    private_agent = _take(headers, "VPA", -1, r)
    government_market = _take(headers, "VDGM", -1, r) + _take(headers, "VIGM", -1, r)
    government_agent = _take(headers, "VGA", -1, r)
    firm_tax = use.to_numpy() - (_take(headers, "VDFM", -1, r)
                                 + _take(headers, "VIFM", -1, r))
    product_taxes = (private_agent - private_market + government_agent - government_market
                     + firm_tax.sum(axis=1))
    investment = (_take(headers, "VINV", -1, r) if "VINV" in headers
                  else np.zeros(len(commodities)))

    fob_imports = _take(headers, "VXWD", 1, r).sum(axis=1)
    cif_imports = _take(headers, "VIWS", -1, r).sum(axis=1)
    market_imports = _take(headers, "VIMS", -1, r).sum(axis=1)
    tariffs = market_imports - cif_imports
    route = _take(headers, "VTWR", -1, r).sum(axis=2)
    services = _labels(headers, "VST", 0)
    modes = _labels(headers, "VTWR", 0)
    service_supply = _take(headers, "VST", -1, r)
    margins = pd.DataFrame(0.0, index=services, columns=commodities)
    used = np.zeros(len(services))
    for i, mode in enumerate(modes):
        service = mode if mode in services else services[i]
        margins.loc[service] += route[i]
        used[i if mode not in services else services.index(mode)] += route[i].sum()
    for service in services:
        margins.loc[service, service] -= used[services.index(service)]
    net_transport_exports = service_supply - used
    exports = _take(headers, "VXMD", 1, r).sum(axis=1) + net_transport_exports
    final = pd.DataFrame({"hh": private_agent, "gov": government_agent,
                          "s_i": investment, "row": exports}, index=commodities)
    return SupplyUse(
        make=make, use=use, final=final,
        imports=pd.Series(fob_imports, index=commodities),
        value_added=value_added,
        product_taxes=pd.Series(product_taxes, index=commodities),
        production_taxes=pd.Series(_take(headers, "OSEP", -1, r), index=activities),
        margins=margins, tariffs=pd.Series(tariffs, index=commodities),
        final_roles={"hh": "households", "gov": "government",
                     "s_i": "investment", "row": "exports"},
        notes={"source": "GTAP", "region": str(region),
               "imports": "FOB; CIF–FOB in margins; VIMS–VIWS in tariffs",
               "transport": "VST less route margins is net service exports"})


def gtap_sam(headers, region=0, **kwargs):
    """Build a balanced one-region SAM from GTAP headers."""
    return sut_to_sam(gtap_supply_use(headers, region), **kwargs)
