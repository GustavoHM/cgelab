"""GTAP route margins and global transport-pool identities."""

import numpy as np
import pandas as pd
import pytest

from cgelab.transport import TransportMargins


def margins():
    fob = np.array([[[80.0], [20.0]], [[40.0], [60.0]]])
    transport = np.array([[[8.0], [4.0]], [[2.0], [6.0]]])
    supply = np.array([[12.0], [8.0]])
    return TransportMargins(("r1", "r2"), ("c",), ("r1", "r2"),
                            fob, transport, supply)


def test_base_year_reproduces_fob_cif_and_global_service_supply():
    margins_ = margins()
    result = margins_.evaluate(margins_.fob, np.ones_like(margins_.fob),
                               np.ones_like(margins_.supply))
    np.testing.assert_allclose(result["vxwd"], margins_.fob)
    np.testing.assert_allclose(result["vtwr"], margins_.transport)
    np.testing.assert_allclose(result["viws"], margins_.cif)
    np.testing.assert_allclose(result["qst"], margins_.supply)
    assert result["qt"] == pytest.approx(margins_.transport.sum())
    assert result["pt"] == pytest.approx(1.0)
    assert result["viws"].sum() == pytest.approx(220.0)


def test_cobb_douglas_pool_changes_service_demand_and_preserves_margin_value():
    margins_ = margins()
    service_prices = np.array([[2.0], [1.0]])
    result = margins_.evaluate(margins_.fob, np.ones_like(margins_.fob), service_prices)
    assert result["pt"] == pytest.approx(2.0 ** 0.6)
    assert result["qst"][0, 0] < margins_.supply[0, 0]
    assert result["qst"][1, 0] > margins_.supply[1, 0]
    assert result["vtwr"].sum() == pytest.approx(result["pt"] * result["qt"])
    assert (service_prices * result["qst"]).sum() == pytest.approx(result["vtwr"].sum())


def test_route_transport_demand_is_proportional_to_shipments():
    margins_ = margins()
    base = margins_.evaluate(margins_.fob, np.ones_like(margins_.fob),
                             np.ones_like(margins_.supply))
    doubled = margins_.evaluate(2.0 * margins_.fob, np.ones_like(margins_.fob),
                                np.ones_like(margins_.supply))
    assert doubled["qt"] == pytest.approx(2.0 * base["qt"])
    np.testing.assert_allclose(doubled["vtwr"], 2.0 * base["vtwr"])


def test_long_reader_validates_and_fills_unlisted_routes():
    routes = pd.DataFrame([
        ("r1", "r1", "c", 80.0, 8.0),
        ("r1", "national", "c", 20.0, 4.0),
        ("r2", "r1", "c", 40.0, 2.0),
        ("r2", "national", "c", 60.0, 6.0),
    ], columns=["origin", "destination", "commodity", "fob", "transport"])
    supply = pd.DataFrame([("r1", "c", 12.0), ("r2", "c", 8.0)],
                          columns=["origin", "commodity", "value"])
    result = TransportMargins.from_long(
        routes, supply, regions=("r1", "r2"), commodities=("c",),
        destinations=("r1", "national"))
    assert result.transport.shape == (2, 2, 1)
    assert result.transport[0, 1, 0] == 4.0
    assert result.transport[1, 0, 0] == 2.0
    np.testing.assert_allclose(result.fob_share + result.transport_share, 1.0)


def test_invalid_transport_data_is_refused():
    margins_ = margins()
    with pytest.raises(ValueError, match="positive fob"):
        TransportMargins(margins_.regions, margins_.commodities, margins_.destinations,
                         np.zeros_like(margins_.fob), margins_.transport, margins_.supply)
    with pytest.raises(ValueError, match="equal global service supply"):
        TransportMargins(margins_.regions, margins_.commodities, margins_.destinations,
                         margins_.fob, margins_.transport, 2.0 * margins_.supply)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        margins_.evaluate(margins_.fob, np.ones_like(margins_.fob),
                          np.array([[np.nan], [1.0]]))
