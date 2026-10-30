"""Two identical regions, written out so the base year is known on paper.

Each region produces one good with labour and capital, sells some of it to the
other region's household, and trades a balanced slice with the rest of the world.
Government is national and is paid a slice of each region's capital.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibrate import calibrate
from ..sam import SAM
from ..spec import ModelSpec
from ..transport import TransportMargins

__all__ = ["build", "build_with_transport", "spec", "ENTRIES", "ASYMMETRIC_IMPORTS"]


def spec(**overrides) -> ModelSpec:
    kwargs = dict(
        regions=("r1", "r2"),
        activities=["r1:a", "r2:a"],
        commodities=["r1:c", "r2:c"],
        households=["r1:hh", "r2:hh"],
        factors=("r1:f_lab", "r1:f_cap", "r2:f_lab", "r2:f_cap"),
        numeraire_factor="r1:f_lab",
        government="gov", investment="s_i", rest_of_world="row",
        sigma_va=1.0, sigma_armington=2.0, sigma_interregional=2.0,
        margins=False,
    )
    kwargs.update(overrides)
    return ModelSpec(**kwargs)


ENTRIES = [
    ("r1:f_lab", "r1:a", 60.0), ("r1:f_cap", "r1:a", 40.0), ("r1:a", "r1:c", 100.0),
    ("r2:f_lab", "r2:a", 60.0), ("r2:f_cap", "r2:a", 40.0), ("r2:a", "r2:c", 100.0),
    ("row", "r1:c", 10.0), ("row", "r2:c", 10.0),
    ("r1:c", "r1:hh", 50.0), ("r1:c", "r2:hh", 20.0),
    ("r1:c", "gov", 10.0), ("r1:c", "s_i", 20.0), ("r1:c", "row", 10.0),
    ("r2:c", "r2:hh", 50.0), ("r2:c", "r1:hh", 20.0),
    ("r2:c", "gov", 10.0), ("r2:c", "s_i", 20.0), ("r2:c", "row", 10.0),
    ("r1:hh", "r1:f_lab", 60.0), ("r1:hh", "r1:f_cap", 30.0), ("gov", "r1:f_cap", 10.0),
    ("r2:hh", "r2:f_lab", 60.0), ("r2:hh", "r2:f_cap", 30.0), ("gov", "r2:f_cap", 10.0),
    ("s_i", "r1:hh", 20.0), ("s_i", "r2:hh", 20.0),
]


#: Moves 5 of imports from r2's account to r1's: r1's household buys 5 more of
#: r1's good and saves 5 less, and investment buys 5 less of r2's good. The two
#: regions then import different shares of the same good.
ASYMMETRIC_IMPORTS = [("row", "r1:c", 5.0), ("r1:c", "r1:hh", 5.0), ("s_i", "r1:hh", -5.0),
                      ("row", "r2:c", -5.0), ("r2:c", "s_i", -5.0)]


def build(model_spec: ModelSpec | None = None, *, rename=None,
          asymmetric_imports: bool = False):
    """Balanced two-region SAM and its calibrated economy.

    ``rename`` is a pair ``(old, new)`` applied to both region labels, so the
    same matrix can be solved under swapped names. ``asymmetric_imports``
    applies :data:`ASYMMETRIC_IMPORTS`.
    """
    rows = ENTRIES + (ASYMMETRIC_IMPORTS if asymmetric_imports else [])
    if rename is not None:
        a, b = rename

        def swap(name: str) -> str:
            if name.startswith(a + ":"):
                return b + ":" + name[len(a) + 1:]
            if name.startswith(b + ":"):
                return a + ":" + name[len(b) + 1:]
            return name

        rows = [(swap(r), swap(c), v) for r, c, v in rows]
    sam = SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]), name="two-region")
    sam.check_balance(abs_tol=1e-9)
    econ = calibrate(sam, model_spec or spec())
    return sam, econ


def build_with_transport(*, regional_investment: bool = False, service_exports: bool = False,
                         **spec_overrides):
    """Two goods and two regions with hand-written route and service margins.

    ``regional_investment`` gives each region its own investment account, so
    investment buys at destination prices that differ in the base year.
    ``service_exports`` has region r1 export 2 of its transport service as
    well as supplying the pool, paid for by 2 more imports of r1's good.
    ``spec_overrides`` go to :func:`spec` (``export_rule``, ``sigma_cet``...).
    """
    entries = list(ENTRIES)
    for region in ("r1", "r2"):
        entries += [
            (f"{region}:f_lab", f"{region}:a_m", 8.0),
            (f"{region}:f_cap", f"{region}:a_m", 4.0),
            (f"{region}:a_m", f"{region}:m", 12.0),
            (f"{region}:m", f"{region}:c", 12.0),
            (f"{region}:m", f"{region}:m", -12.0),
            (f"{region}:c", f"{region}:hh", 12.0),
            (f"{region}:hh", f"{region}:f_lab", 8.0),
            (f"{region}:hh", f"{region}:f_cap", 4.0),
        ]
    if service_exports:
        entries += [("r1:f_lab", "r1:a_m", 2.0), ("r1:a_m", "r1:m", 2.0), ("r1:m", "row", 2.0),
                    ("row", "r1:c", 2.0), ("r1:c", "r1:hh", 2.0), ("r1:hh", "r1:f_lab", 2.0)]
    investment = {}
    if regional_investment:
        def split(name, other):
            return f"{other.split(':')[0]}:s_i" if name == "s_i" else name
        entries = [(split(r, c), split(c, r), v) for r, c, v in entries]
        investment = dict(investment="r1:s_i", regional_investment={"r2": "r2:s_i"})
    sam = SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]),
                        name="two-region transport")
    sam.check_balance(abs_tol=1e-9)
    model_spec = spec(
        activities=[f"{r}:{a}" for r in ("r1", "r2") for a in ("a", "a_m")],
        commodities=[f"{r}:{c}" for r in ("r1", "r2") for c in ("c", "m")],
        margins=True, **investment, **spec_overrides)
    fob = np.zeros((2, 3, 2))
    fob[0, :, 0] = [45.0, 18.0, 27.0]
    # With its own account, r2's investment in its own good leaves the
    # national channel for destination r2.
    fob[1, :, 0] = [18.0, 65.0, 7.0] if regional_investment else [18.0, 45.0, 27.0]
    transport = np.zeros_like(fob)
    transport[0, 0, 0] = 12.0
    transport[1, 1, 0] = 12.0
    supply = np.array([[0.0, 12.0], [0.0, 12.0]])
    margins = TransportMargins(("r1", "r2"), ("c", "m"),
                               ("r1", "r2", "national"), fob, transport, supply)
    return sam, calibrate(sam, model_spec, transport=margins)
