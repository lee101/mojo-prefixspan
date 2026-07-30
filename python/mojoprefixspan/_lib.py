"""ctypes bridge to the Mojo PrefixSpan kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PREFIXSPAN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-prefixspan.so"
)
SRC = os.path.join(ROOT, "src", "prefixspan.mojo")
I = ctypes.c_int64
I64_ARRAY = np.ctypeslib.ndpointer(
    dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE")
)

_lib: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_PREFIXSPAN_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        fn = _lib.mps_mine
        fn.argtypes = [
            I64_ARRAY,
            I64_ARRAY,
            I,
            I,
            I,
            I,
            I,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I,
        ]
        fn.restype = None
        topk_fn = _lib.mps_topk
        topk_fn.argtypes = [
            I64_ARRAY,
            I64_ARRAY,
            I,
            I,
            I,
            I,
            I,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I64_ARRAY,
            I,
        ]
        topk_fn.restype = None
    return _lib


def checked_i64(value: int, name: str) -> int:
    """Convert a Python integer without allowing ctypes to narrow it silently."""
    value = int(value)
    if not -(1 << 63) <= value < (1 << 63):
        raise OverflowError(f"{name} does not fit in a signed 64-bit integer")
    return value
