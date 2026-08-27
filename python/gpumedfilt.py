"""NumPy front end for the gpumedfilt shared library (2D median filter on the GPU).

    import numpy as np
    from gpumedfilt import medfilt2

    img = np.random.rand(1080, 1920).astype(np.float32)
    out = medfilt2(img, 5)            # float32 array, same shape

The library is located through, in order: the GPUMEDFILT_LIBRARY environment variable, the
directory of this file, and the CMake build directories next to it (build/Release, build).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

__all__ = ["medfilt2", "load_library"]

_LIB_NAMES = {
    "win32": ["gpumedfilt.dll"],
    "darwin": ["libgpumedfilt.dylib"],
}.get(sys.platform, ["libgpumedfilt.so"])

_lib: ctypes.CDLL | None = None


def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    roots = [here, here.parent / "build" / "Release", here.parent / "build",
             here.parent / "build" / "Debug", Path.cwd()]
    paths = []
    env = os.environ.get("GPUMEDFILT_LIBRARY")
    if env:
        paths.append(Path(env))
    for root in roots:
        for name in _LIB_NAMES:
            paths.append(root / name)
    return paths


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    fn = lib.gpu_medfilt2
    fn.restype = None
    fn.argtypes = [
        ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),           # input
        ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS,WRITEABLE"),  # output
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # filter_size
    ]
    return lib


def load_library(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """Load the shared library (done automatically on first use; call explicitly to pick a file).

    Without `path`, the fixed locations from _candidate_paths() are tried first, then the system
    loader's own search (LD_LIBRARY_PATH / ldconfig cache on Linux, DYLD paths on macOS, PATH on
    Windows) for an installed copy.
    """
    global _lib
    candidates = [Path(path)] if path is not None else _candidate_paths()
    errors = []
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            _lib = _bind(ctypes.CDLL(str(cand)))
            return _lib
        except (OSError, AttributeError) as e:  # wrong architecture, missing dependency, no such symbol
            errors.append(f"{cand}: {e}")
    if path is None:
        names = list(_LIB_NAMES)
        found = ctypes.util.find_library("gpumedfilt")
        if found:
            names.insert(0, found)
        for name in names:
            try:
                _lib = _bind(ctypes.CDLL(name))
                return _lib
            except (OSError, AttributeError) as e:
                errors.append(f"{name} (system search): {e}")
    tried = "\n  ".join(str(c) for c in candidates)
    detail = ("\n" + "\n".join(errors)) if errors else ""
    raise OSError(
        "gpumedfilt shared library not found. Build it with CMake, install it, or set GPUMEDFILT_LIBRARY. "
        f"Tried:\n  {tried}{detail}"
    )


def medfilt2(image: np.ndarray, filter_size: int, out: np.ndarray | None = None) -> np.ndarray:
    """Median-filter a 2D image on the GPU.

    Parameters
    ----------
    image : array_like, shape (height, width)
        Any real dtype; converted to float32 (a copy is made unless it already is C-contiguous float32).
    filter_size : int
        Odd window size (filter_size x filter_size). Borders are reflected symmetrically
        (d c b a | a b c d), so it may exceed the image size.
    out : ndarray, optional
        C-contiguous float32 array of the same shape to write into; must not alias `image`.

    Returns
    -------
    ndarray of float32, shape (height, width)
    """
    lib = _lib or load_library()

    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D (height, width), got shape {img.shape}")
    height, width = img.shape
    if width == 0 or height == 0:
        raise ValueError("image must not be empty")
    if max(width, height) > 2**31 - 1 or width * height * 4 > 2**63 - 1:
        raise ValueError("image too large")
    filter_size = int(filter_size)
    if filter_size < 1 or filter_size % 2 == 0:
        raise ValueError(f"filter_size must be a positive odd integer, got {filter_size}")

    if out is None:
        out = np.empty_like(img)
    else:
        if out.shape != img.shape or out.dtype != np.float32 or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out must be a writeable C-contiguous float32 array with the image's shape")
        if np.shares_memory(out, img):
            raise ValueError("out must not share memory with image")

    lib.gpu_medfilt2(img, out, width, height, filter_size)
    return out


def _reference(image: np.ndarray, k: int) -> np.ndarray:
    """Pure-NumPy reference with the same symmetric border, for the self-test."""
    r = k // 2
    padded = np.pad(image, r, mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return np.median(windows.reshape(*image.shape, k * k), axis=-1).astype(np.float32)


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    lib = load_library()
    print(f"loaded {lib._name}")

    ok = True
    for (h, w), k in [((1, 1), 3), ((7, 5), 3), ((7, 5), 9), ((64, 100), 5), ((131, 257), 7),
                      ((131, 257), 15), ((40, 64), 31), ((40, 64), 101)]:
        img = rng.random((h, w), dtype=np.float32)
        got = medfilt2(img, k)
        ref = _reference(img, k)
        good = np.array_equal(got, ref)
        ok &= good
        print(f"  {w}x{h} K={k}: {'ok' if good else 'MISMATCH'}")

    # a non-float32, non-contiguous input is converted transparently
    img64 = rng.random((50, 60)).T
    good = np.array_equal(medfilt2(img64, 5), _reference(img64.astype(np.float32), 5))
    ok &= good
    print(f"  float64 transposed input: {'ok' if good else 'MISMATCH'}")

    img = rng.random((2160, 4096), dtype=np.float32)
    #medfilt2(img, 201)
    t0 = time.perf_counter()
    for _ in range(5):
        medfilt2(img, 201)
    dt = (time.perf_counter() - t0) / 5
    print(f"  4096x2160 K=201: {dt * 1e3:.1f} ms per call including transfers")

    print("all ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)
