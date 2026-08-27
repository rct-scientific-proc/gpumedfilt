# gpumedfilt

2D median filter for single-precision images on NVIDIA GPUs, built as a shared library with a C API.

```c
#include "gpumedfilt.h"
// input/output: row-major float arrays in CPU memory; filter_size: any odd number >= 1.
// Borders are handled by symmetric reflection (d c b a | a b c d), which also works for
// filter sizes larger than the image.
gpu_medfilt2(input, output, width, height, filter_size);
```

Errors (bad arguments, CUDA failures) are reported on `stderr`; the output buffer is left untouched.

## Build

Requires CMake ≥ 3.24, a CUDA toolkit (tested with 13.3) and a host compiler (MSVC on Windows).

```sh
cmake -S . -B build              # targets the GPU in this machine; -DCMAKE_CUDA_ARCHITECTURES="86;89" to override
cmake --build build --config Release
ctest --test-dir build -C Release   # bit-exact comparison against a CPU reference
```

Produces `gpumedfilt.dll`/`.lib` (or `libgpumedfilt.so`) with the CUDA runtime linked statically, and
`medfilt_test`, which also benchmarks the kernels (`medfilt_test --bench`).

## Python

[python/gpumedfilt.py](python/gpumedfilt.py) wraps the library with `ctypes` (NumPy is the only dependency):

```python
import numpy as np
from gpumedfilt import medfilt2          # finds build/Release/gpumedfilt.dll, or set GPUMEDFILT_LIBRARY

out = medfilt2(image, 5)                 # any real 2-D array -> float32 result of the same shape
medfilt2(image, 5, out=buffer)           # reuse a float32 output buffer: saves ~3 ms per 4K frame
```

`python gpumedfilt.py` runs a self-test against a NumPy reference (`np.pad(mode="symmetric")` + `np.median`).

## Implementation

Values are processed as order-preserving 32-bit integer keys of the float bit patterns, which makes
compare/min/max single integer instructions and gives an exact result for every input, including
±0, ±inf and NaN (a total order in which NaNs sort above +inf).

| filter size | kernel |
|---|---|
| 3, 5, 7, 9 | Column segments are sorted once into shared memory (optimal sorting networks) and shared by the K windows that contain them. Per pixel: sort the K rows in registers, drop the elements that provably cannot be the median of a row/column-sorted matrix, and select the median of the remaining ones (13 of 25, 29 of 49, 47 of 81) by forgetful selection. |
| 11 … ~99 | A 128-thread block walks a 128-pixel-wide strip down the image. Sorted column segments are kept in shared memory and updated incrementally (one value out, one in) per row. Per pixel the median is found by bisection on the key value: count(≤ probe) is a binary search in each sorted column (8 columns searched in lockstep for latency hiding); the bracket is snapped to actual window elements after every probe, so it converges in about log2(#distinct values) steps. |
| ~100 … ~1000 | Same scheme with the sorted columns in a global-memory scratch buffer (persistent blocks, one scratch slot each). Shared memory holds every G-th element of every column (G = 8…64), so each probe does the binary search in shared memory and then reads a single G-element group (one or a few 32-byte sectors) per column from global memory. |
| larger | Same bisection reading the raw window directly from global memory (no size limit). |

Kernel times on an RTX 4070 for a 4096×2160 image (host↔device copies of 35 MB each add ~6.5 ms
in total at pageable-memory PCIe speed and dominate the end-to-end time for small filters):

| K | 3 | 5 | 7 | 9 | 11 | 15 | 21 | 31 | 63 | 101 | 201 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ms | 0.17 | 0.31 | 0.86 | 1.9 | 3.0 | 4.8 | 7.0 | 10.5 | 28 | 160 | 280 |

(8-bit-like content; uniformly random floats are 10–110 % slower at K ≥ 11 because they need more
bisection steps: 0.59 s instead of 0.28 s at K = 201.)

NPP's `nppiFilterMedianBorder_32f_C1R` was evaluated as an alternative: it is no faster for 3×3/5×5,
its border modes do not implement symmetric reflection, and on this toolkit (13.3) it fails with an
illegal memory access for masks ≥ 7×7 on 4K images, so it is not used.
