/* Private API shared between gpumedfilt.cu and the test/benchmark harness.
 * Not part of the public interface; not exported from the shared library. */
#ifndef GPUMEDFILT_INTERNAL_H
#define GPUMEDFILT_INTERNAL_H

#include <cuda_runtime.h>

enum gpumedfilt_algo {
    GPUMEDFILT_ALGO_AUTO       = 0,  /* dispatch on filter_size (what gpu_medfilt2 uses) */
    GPUMEDFILT_ALGO_SMALL      = 1,  /* sorted columns + row sort + candidate pruning in registers (K = 3,5,7,9) */
    GPUMEDFILT_ALGO_BISECT_128 = 2,  /* incremental sorted columns + snapping bisection, 128-wide strip */
    GPUMEDFILT_ALGO_BISECT_256 = 3,  /* same, 256-wide strip */
    GPUMEDFILT_ALGO_GLOBAL     = 4,  /* no shared memory, any K (fallback) */
    GPUMEDFILT_ALGO_COUNT      = 5
};

/* Median filter on device buffers (row-major, width*height floats each), asynchronous on `stream`.
 * Returns cudaErrorInvalidValue if `algo` cannot handle this filter_size (e.g. shared memory too small). */
cudaError_t gpumedfilt_device(const float* d_in, float* d_out, int width, int height,
                              int filter_size, cudaStream_t stream, int algo);

const char* gpumedfilt_algo_name(int algo);
int gpumedfilt_auto_algo(int filter_size);   /* the algo GPUMEDFILT_ALGO_AUTO resolves to */

#endif
