"""har_harpy.py — the HAR reader against harpy3, header by header, on a GTAP year folder.

harpy3 (GEMPACK's reader, GPLv3) is NOT a dependency of cgelab. Install it in an
environment of its own; it needs a NumPy that still has ``np.chararray``:

    python -m venv /tmp/harpyenv
    /tmp/harpyenv/bin/pip install harpy3 "numpy<2.3"
    HARPY_PYTHON=/tmp/harpyenv/bin/python python validation/har_harpy.py /path/to/GTAP10A/GTAP/2014

harpy3 does not read type RL, so files holding one (co2.har, gsdvole.har) are left
out of the comparison; cgelab reads them. On GTAP 10A, 2014, the 86 headers of
sets.har, basedata.har, default.prm and metadata.har agree exactly: values,
set names and element labels.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import cgelab as cge

FILES = ("sets.har", "basedata.har", "default.prm", "metadata.har")

DUMP = r'''
import sys, pickle, numpy as np
from harpy import HarFileObj
out = {}
for fn in sys.argv[2:]:
    hf = HarFileObj.loadFromDisk(filename=fn)
    for name in hf.getHeaderArrayNames():
        ha = hf.getHeaderArrayObj(name)
        arr = np.asarray(ha["array"])
        if arr.dtype.kind in "SUO":
            arr = np.array([str(x.decode() if isinstance(x, bytes) else x).rstrip()
                            for x in arr.ravel()])
        sets = [(s["name"], list(s["dim_desc"]) if s["dim_desc"] is not None else None)
                for s in (ha.get("sets") or [])]
        out[(fn.split("/")[-1], name.strip())] = (arr, sets)
pickle.dump(out, open(sys.argv[1], "wb"))
'''


def main() -> int:
    if len(sys.argv) < 2 or "HARPY_PYTHON" not in os.environ:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        script, dump = Path(tmp) / "dump.py", Path(tmp) / "harpy.pkl"
        script.write_text(DUMP)
        subprocess.run([os.environ["HARPY_PYTHON"], "-I", str(script), str(dump),
                        *[str(folder / f) for f in FILES]], check=True,
                       stderr=subprocess.DEVNULL)
        ref = pickle.loads(dump.read_bytes())
    ours = {f: cge.read_har(folder / f) for f in FILES}
    worst, bad = 0.0, []
    for (fn, name), (arr, sets) in ref.items():
        h = ours[fn][name]
        if arr.dtype.kind == "U":
            if list(np.asarray(h.array).ravel()) != list(arr.ravel()):
                bad.append((fn, name, "text"))
            continue
        a = np.asarray(h.array, float)
        b = np.asarray(arr, float).reshape(a.shape)
        worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
        if sets and ([s for s, _ in sets] != h.sets
                     or any(l is not None and list(l) != hl for (_, l), hl in zip(sets, h.labels))):
            bad.append((fn, name, "labels"))
    print(f"{len(ref)} headers compared; worst absolute difference {worst:g}; "
          f"label or text mismatches: {bad or 'none'}")
    return 0 if worst == 0.0 and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
