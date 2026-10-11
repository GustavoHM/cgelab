"""Two identical regions, written out so the base year is known on paper.

Each region produces one good with labour and capital, sells some of it to the
other region's household, and trades a balanced slice with the rest of the world.
Government is national and is paid a slice of each region's capital.
"""

from __future__ import annotations

import pandas as pd

from ..calibrate import calibrate
from ..sam import SAM
from ..spec import ModelSpec

__all__ = ["build", "spec", "ENTRIES"]


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


def build(model_spec: ModelSpec | None = None, *, rename=None):
    """Balanced two-region SAM and its calibrated economy.

    ``rename`` is a pair ``(old, new)`` applied to both region labels, so the
    same matrix can be solved under swapped names.
    """
    rows = ENTRIES
    if rename is not None:
        a, b = rename

        def swap(name: str) -> str:
            if name.startswith(a + ":"):
                return b + ":" + name[len(a) + 1:]
            if name.startswith(b + ":"):
                return a + ":" + name[len(b) + 1:]
            return name

        rows = [(swap(r), swap(c), v) for r, c, v in ENTRIES]
    sam = SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]), name="two-region")
    sam.check_balance(abs_tol=1e-9)
    econ = calibrate(sam, model_spec or spec())
    return sam, econ
