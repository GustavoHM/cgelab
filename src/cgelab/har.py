"""Read GEMPACK header array (HAR) files, the format of the GTAP database.

A HAR file is a sequence of Fortran unformatted records, each framed by its
length in a little-endian 32-bit integer. Every header (an array) starts with a
four-character name record and a record giving its type, storage, a 70-character
description and its dimensions; the data follow. Types read here:

* ``1C`` -- character strings (set element lists, notes);
* ``2I`` / ``2R`` -- two-dimensional integers or reals without set labels;
* ``RE`` -- reals of up to seven dimensions with a set name and element labels
  per dimension, stored ``FULL`` (in slices) or ``SPSE`` (index-value pairs);
* ``RL`` -- the same without the set record; trailing dimensions of size one
  are dropped.

Values are single precision in the file and come back as ``float64``. The data
are column-major, as Fortran writes them, so the first index runs fastest.

The reader is written from the files themselves and checked two ways in the
tests: against the GTAP accounting identities, which only close when every
index lands where it should, and against an independent reader.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Header", "read_har", "write_har"]


@dataclass
class Header:
    """One array from a HAR file."""

    name: str
    description: str
    kind: str
    array: np.ndarray
    #: Set name per dimension, and its element labels. Empty for unlabelled types.
    sets: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def to_series(self) -> pd.Series:
        """Long form: one entry per element, indexed by the labels (non-zeros kept)."""
        if not self.sets:
            raise ValueError(f"{self.name} carries no set labels")
        index = pd.MultiIndex.from_product(self.labels, names=self.sets)
        return pd.Series(self.array.ravel(order="C"), index=index, name=self.name)

    def __repr__(self) -> str:
        dims = " x ".join(f"{s}[{len(l)}]" for s, l in zip(self.sets, self.labels)) or str(
            self.array.shape)
        return f"Header({self.name!r}, {self.kind}, {dims}: {self.description!r})"


def _records(data: bytes):
    pos, n = 0, len(data)
    while pos < n:
        (length,) = struct.unpack_from("<i", data, pos)
        start = pos + 4
        end = start + length
        if end + 4 > n:
            raise ValueError(f"truncated record at byte {pos}")
        (trailer,) = struct.unpack_from("<i", data, end)
        if trailer != length:
            raise ValueError(f"record at byte {pos}: length {length}, trailer {trailer}")
        yield data[start:end]
        pos = end + 4


def _ints(rec: bytes, offset: int, count: int) -> tuple:
    return struct.unpack_from(f"<{count}i", rec, offset)


def _text(rec: bytes, start: int, width: int, count: int) -> list:
    return [rec[start + i * width:start + (i + 1) * width].decode("latin-1").rstrip()
            for i in range(count)]


def _read_strings(recs, it) -> list:
    out = []
    while True:
        rec = next(it)
        remaining, total, here = _ints(rec, 4, 3)
        width = (len(rec) - 16) // here if here else 0
        out += _text(rec, 16, width, here)
        if remaining <= 1:
            return out


def _read_2d(rec_iter, dims, dtype):
    d1, d2 = dims
    out = np.zeros(d1 * d2, dtype=dtype)
    filled = 0
    while filled < d1 * d2:
        rec = next(rec_iter)
        remaining, n1, n2, a1, b1, a2, b2 = _ints(rec, 4, 7)
        count = (b1 - a1 + 1) * (b2 - a2 + 1)
        values = np.frombuffer(rec, dtype="<f4" if dtype == float else "<i4",
                               count=count, offset=32)
        block = values.reshape((b2 - a2 + 1, b1 - a1 + 1)).T
        full = out.reshape((d2, d1)).T
        full[a1 - 1:b1, a2 - 1:b2] = block
        filled += count
    return out.reshape((d2, d1)).T


def _read_re(it, dims, storage, labelled=True):
    if not labelled:
        set_names, labels_by_set, coefficient = [], {}, ""
    else:
        set_names, labels_by_set, coefficient = _read_set_info(it)
    return _read_real_body(it, dims, storage, set_names, labels_by_set, coefficient)


def _read_set_info(it):
    rec = next(it)
    # Distinct sets, a flag, and the number of dimensions that carry a set: a
    # TRAD_COMM x REG x REG array has two distinct sets over three dimensions,
    # and each distinct set's labels are written once.
    _distinct, _, nsets = _ints(rec, 4, 3)
    coefficient = rec[16:28].decode("latin-1").rstrip()
    set_names = _text(rec, 32, 12, nsets)
    flags = rec[32 + 12 * nsets:32 + 13 * nsets].decode("latin-1")
    labels_by_set: dict = {}
    for name, flag in zip(set_names, flags):
        if flag == "k" and name not in labels_by_set:
            labels_by_set[name] = _read_strings(None, it)
    return set_names, labels_by_set, coefficient


def _read_real_body(it, dims, storage, set_names, labels_by_set, coefficient):
    ndim = len(dims)
    shape = tuple(dims)
    if storage == "FULL":
        head = next(it)                         # remaining, 7, dims
        size = int(np.prod(shape))
        flat = np.zeros(size, dtype=np.float64)
        cube = flat.reshape(tuple(reversed(shape))).transpose(tuple(range(ndim - 1, -1, -1)))
        while True:
            rec = next(it)
            remaining, *bounds = _ints(rec, 4, 15)
            lo = bounds[0::2]
            hi = bounds[1::2]
            data = next(it)
            data_remaining = _ints(data, 4, 1)[0]
            sl = tuple(slice(a - 1, b) for a, b in zip(lo, hi))
            ext = tuple(b - a + 1 for a, b in zip(lo, hi))
            count = int(np.prod(ext))
            values = np.frombuffer(data, dtype="<f4", count=count, offset=8)
            cube[sl] = values.reshape(tuple(reversed(ext))).transpose(
                tuple(range(ndim - 1, -1, -1)))
            if data_remaining <= 1:
                break
        array = cube
    elif storage == "SPSE":
        next(it)                                # remaining, sizes, filler
        size = int(np.prod(shape))
        flat = np.zeros(size, dtype=np.float64)
        while True:
            rec = next(it)
            remaining, total, here = _ints(rec, 4, 3)
            idx = np.frombuffer(rec, dtype="<i4", count=here, offset=16)
            vals = np.frombuffer(rec, dtype="<f4", count=here, offset=16 + 4 * here)
            flat[idx - 1] = vals
            if remaining <= 1:
                break
        array = flat.reshape(tuple(reversed(shape))).transpose(tuple(range(ndim - 1, -1, -1)))
    else:
        raise ValueError(f"unknown storage {storage!r}")
    used = len(set_names)
    if not used:                       # unlabelled: drop the padding to seven dimensions
        keep = len(shape)
        while keep > 1 and shape[keep - 1] == 1:
            keep -= 1
        used = keep
    array = np.ascontiguousarray(array.reshape(shape[:used]))
    labels = [labels_by_set.get(s, [str(i + 1) for i in range(shape[k])])
              for k, s in enumerate(set_names)]
    return array, set_names, labels, coefficient


def read_har(path, headers=None) -> dict:
    """Read a HAR file into ``{name: Header}``.

    ``headers`` restricts the result to some names (the file is still scanned
    once). Unsupported types raise, naming the header, rather than being
    skipped: a silently missing array would surface later as a wrong total.
    """
    data = Path(path).read_bytes()
    it = _records(data)
    wanted = None if headers is None else {h.strip() for h in headers}
    out = {}
    while True:
        try:
            rec = next(it)
        except StopIteration:
            break
        if len(rec) != 4:
            raise ValueError(f"expected a 4-character header name, got a {len(rec)}-byte record")
        name = rec.decode("latin-1").strip()
        info = next(it)
        kind = info[4:6].decode("latin-1")
        storage = info[6:10].decode("latin-1")
        desc = info[10:80].decode("latin-1").strip()
        (ndim,) = _ints(info, 80, 1)
        dims = _ints(info, 84, ndim)
        if kind == "1C":
            strings = _read_strings(None, it)
            arr = np.array(strings, dtype=object)
            header = Header(name, desc, kind, arr)
        elif kind in ("2I", "2R"):
            arr = _read_2d(it, dims, int if kind == "2I" else float)
            header = Header(name, desc, kind, arr.astype(np.float64 if kind == "2R" else np.int64))
        elif kind in ("RE", "RL"):
            arr, sets, labels, _ = _read_re(it, dims, storage, labelled=kind == "RE")
            header = Header(name, desc, kind, arr, sets, labels)
        else:
            raise ValueError(f"header {name!r}: type {kind!r} is not supported")
        if wanted is None or name in wanted:
            out[name] = header
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _frame(payload: bytes) -> bytes:
    n = struct.pack("<i", len(payload))
    return n + payload + n


def _info(kind: str, storage: str, desc: str, dims) -> bytes:
    desc = desc[:70].ljust(70).encode("latin-1")
    return (b"    " + kind.encode() + storage.encode() + desc
            + struct.pack(f"<i{len(dims)}i", len(dims), *dims))


def _strings_record(strings, width) -> bytes:
    body = b"".join(s[:width].ljust(width).encode("latin-1") for s in strings)
    return _frame(b"    " + struct.pack("<3i", 1, len(strings), len(strings)) + body)


def _header_bytes(name: str, h) -> bytes:
    if len(name) > 4:
        raise ValueError(f"header names have at most four characters, got {name!r}")
    out = _frame(name.ljust(4).encode("latin-1"))
    if isinstance(h, Header):
        kind, desc, array, sets, labels = h.kind, h.description, h.array, h.sets, h.labels
    else:
        array = np.asarray(h)
        if array.dtype.kind in "OUS":
            kind = "1C"
        elif array.dtype.kind in "iu":
            kind = "2I"
        else:
            kind = "2R" if array.ndim <= 2 else "RL"     # 2R is the type every reader takes
        desc, sets, labels = "", [], []
    if kind == "1C":
        strings = [str(x) for x in np.asarray(array, dtype=object).ravel()]
        width = max([len(x) for x in strings] + [1])
        out += _frame(_info("1C", "FULL", desc, (len(strings), width)))
        return out + _strings_record(strings, width)
    if kind in ("2I", "2R"):
        a = np.asarray(array)
        a = a.reshape(a.shape[0], -1) if a.ndim != 2 else a
        d1, d2 = a.shape
        out += _frame(_info(kind, "FULL", desc, (d1, d2)))
        fmt = "<i4" if kind == "2I" else "<f4"
        data = np.asarray(a, dtype=fmt).T.tobytes()
        return out + _frame(b"    " + struct.pack("<7i", 1, d1, d2, 1, d1, 1, d2) + data)
    # RE / RL: reals, one FULL slice
    a = np.asarray(array, dtype=np.float64)
    if a.ndim > 7:
        raise ValueError("a HAR array has at most seven dimensions")
    dims = tuple(a.shape) + (1,) * (7 - a.ndim)
    labelled = bool(sets)
    out += _frame(_info("RE" if labelled else "RL", "FULL", desc, dims))
    if labelled:
        distinct = list(dict.fromkeys(sets))
        info = (b"    " + struct.pack("<3i", len(distinct), -1, len(sets))
                + name.ljust(12).encode("latin-1") + struct.pack("<i", -1)
                + b"".join(s.ljust(12).encode("latin-1") for s in sets)
                + b"k" * len(sets) + b"\x00" * (4 * (len(sets) + 1)))
        out += _frame(info)
        written = set()
        for s, lab in zip(sets, labels):
            if s not in written:
                out += _strings_record(list(lab), 12)
                written.add(s)
    out += _frame(b"    " + struct.pack("<9i", 3, 7, *dims))
    bounds = []
    for d in dims:
        bounds += [1, d]
    out += _frame(b"    " + struct.pack("<15i", 2, *bounds))
    data = np.asarray(a, dtype="<f4").reshape(a.shape).transpose().tobytes()
    return out + _frame(b"    " + struct.pack("<i", 1) + data)


def write_har(path, headers) -> None:
    """Write ``{name: Header or array}`` to a HAR file.

    Reals are written ``FULL`` in one slice, in single precision, as GEMPACK
    stores them. A plain array becomes ``RL`` (reals), ``2I`` (integers) or
    ``1C`` (strings); a :class:`Header` with sets becomes ``RE`` with its labels.
    """
    with open(path, "wb") as f:
        for name, h in headers.items():
            f.write(_header_bytes(name, h))
