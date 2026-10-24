"""Route-level transport margins and the global transport pool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TransportMargins"]


@dataclass
class TransportMargins:
    """Base-year FOB, transport and service-supply values on a regional grid."""

    regions: tuple
    commodities: tuple
    destinations: tuple
    fob: np.ndarray
    transport: np.ndarray
    supply: np.ndarray

    def __post_init__(self):
        self.regions = tuple(self.regions)
        self.commodities = tuple(self.commodities)
        self.destinations = tuple(self.destinations)
        shape = (len(self.regions), len(self.destinations), len(self.commodities))
        self.fob = np.asarray(self.fob, dtype=float)
        self.transport = np.asarray(self.transport, dtype=float)
        self.supply = np.asarray(self.supply, dtype=float)
        if self.fob.shape != shape or self.transport.shape != shape:
            raise ValueError(f"fob and transport must have shape {shape}")
        if self.supply.shape != (len(self.regions), len(self.commodities)):
            raise ValueError(
                f"supply must have shape {(len(self.regions), len(self.commodities))}")
        for name, values in (("fob", self.fob), ("transport", self.transport),
                             ("supply", self.supply)):
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative")
        if np.any((self.fob <= 0) & (self.transport > 0)):
            raise ValueError("a route with positive transport must have positive fob")
        if self.transport.sum() <= 0 or self.supply.sum() <= 0:
            raise ValueError("transport and supply must each have a positive total")
        if not np.isclose(self.transport.sum(), self.supply.sum(), rtol=1e-10, atol=1e-10):
            raise ValueError("global transport margins must equal global service supply")

    @classmethod
    def from_long(cls, routes, supply, *, regions, commodities, destinations=None):
        """Read route rows and a service-supply frame without licensed GTAP data."""
        regions = tuple(regions)
        commodities = tuple(commodities)
        destinations = tuple(destinations or regions)
        shape = (len(regions), len(destinations), len(commodities))
        required = {"origin", "destination", "commodity", "fob", "transport"}
        missing = required - set(routes.columns)
        if missing:
            raise ValueError(f"routes is missing columns {sorted(missing)}")
        if routes.duplicated(["origin", "destination", "commodity"]).any():
            raise ValueError("routes has duplicate origin-destination-commodity rows")
        fob = np.zeros(shape)
        transport = np.zeros(shape)
        for row in routes.itertuples(index=False):
            try:
                index = (regions.index(row.origin), destinations.index(row.destination),
                         commodities.index(row.commodity))
            except ValueError as exc:
                raise ValueError(
                    f"unknown route ({row.origin!r}, {row.destination!r}, {row.commodity!r})") from exc
            fob[index], transport[index] = row.fob, row.transport
        if isinstance(supply, np.ndarray):
            service = supply
        else:
            required = {"origin", "commodity", "value"}
            missing = required - set(supply.columns)
            if missing:
                raise ValueError(f"supply is missing columns {sorted(missing)}")
            if supply.duplicated(["origin", "commodity"]).any():
                raise ValueError("supply has duplicate origin-commodity rows")
            service = np.zeros((len(regions), len(commodities)))
            for row in supply.itertuples(index=False):
                try:
                    index = (regions.index(row.origin), commodities.index(row.commodity))
                except ValueError as exc:
                    raise ValueError(
                        f"unknown service supply ({row.origin!r}, {row.commodity!r})") from exc
                service[index] = row.value
        return cls(regions, commodities, destinations, fob, transport, service)

    @property
    def coefficient(self) -> np.ndarray:
        """Transport services per unit shipped on each route."""
        return np.divide(self.transport, self.fob, out=np.zeros_like(self.transport),
                         where=self.fob > 0)

    @property
    def cif(self) -> np.ndarray:
        return self.fob + self.transport

    @property
    def fob_share(self) -> np.ndarray:
        return np.divide(self.fob, self.cif, out=np.ones_like(self.fob), where=self.cif > 0)

    @property
    def transport_share(self) -> np.ndarray:
        return np.divide(self.transport, self.cif, out=np.zeros_like(self.fob),
                         where=self.cif > 0)

    def evaluate(self, shipments, fob_prices, service_prices) -> dict:
        """Apply the GTAP transport equations at quantities and relative prices."""
        shipments = np.asarray(shipments, dtype=float)
        fob_prices = np.asarray(fob_prices, dtype=float)
        service_prices = np.asarray(service_prices, dtype=float)
        if shipments.shape != self.fob.shape or fob_prices.shape != self.fob.shape:
            raise ValueError(f"shipments and fob_prices must have shape {self.fob.shape}")
        if service_prices.shape != self.supply.shape:
            raise ValueError(f"service_prices must have shape {self.supply.shape}")
        if not np.isfinite(shipments).all() or (shipments < 0).any():
            raise ValueError("shipments must be finite and nonnegative")
        if not np.isfinite(fob_prices).all() or (fob_prices <= 0).any():
            raise ValueError("fob_prices must be finite and positive")
        positive = self.supply > 0
        if (not np.isfinite(service_prices).all() or (service_prices < 0).any()
                or (service_prices[positive] <= 0).any()):
            raise ValueError("service_prices must be finite and nonnegative, and positive "
                             "wherever supply is positive")
        share = self.supply / self.supply.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pt = float(np.prod(np.where(positive, service_prices ** share, 1.0)))
        qt = float((self.coefficient * shipments).sum())
        qst = np.divide(share * pt * qt, service_prices,
                        out=np.zeros_like(self.supply), where=positive)
        vtwr = self.coefficient * shipments * pt
        vxwd = fob_prices * shipments
        return {"qt": qt, "pt": pt, "qst": qst, "vtwr": vtwr,
                "vxwd": vxwd, "viws": vxwd + vtwr}
