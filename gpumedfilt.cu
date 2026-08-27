// 2D median filter for float images on the GPU, with symmetric ("d c b a | a b c d") border handling.
//
// All arithmetic is done on the bit patterns of the floats mapped to unsigned 32-bit keys with the
// same ordering (f2k/k2f below). This makes min/max/compare single integer instructions, gives a
// total order (no NaN special cases, no infinite loops) and lets the result be converted back exactly.
//
// Three kernels, selected by filter size K:
//   * K = 3, 5, 7, 9   : sorted columns in shared memory, then per pixel a row sort, static pruning
//                        of the median candidates and forgetful selection, all in registers
//                        (medfilt_small_kernel; K = 3 reduces to the classic 19-compare-exchange
//                        network).
//   * any other odd K  : sorted columns in shared memory, maintained incrementally as the block walks
//                        down the image, and per pixel a bisection on the key value whose count(<= v)
//                        is a binary search in every sorted column (medfilt_bisect_kernel).
//   * K too large for shared memory (K > ~100): the same bisection straight from global memory.

#include "gpumedfilt.h"
#include "gpumedfilt_internal.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

typedef unsigned int u32;

// ------------------------------------------------------------------------------------------------
// Device helpers
// ------------------------------------------------------------------------------------------------

// Monotone float -> uint32 key: k(a) < k(b) <=> a < b (with -0 < +0 and NaNs at the extremes).
__device__ __forceinline__ u32 f2k(float f)
{
    const u32 u = __float_as_uint(f);
    return u ^ ((u32)(-(int)(u >> 31)) | 0x80000000u);
}

__device__ __forceinline__ float k2f(u32 k)
{
    return __uint_as_float(k ^ ((u32)(-(int)((k >> 31) ^ 1u)) | 0x80000000u));
}

// Symmetric reflection: ... 3 2 1 0 | 0 1 2 ... n-1 | n-1 n-2 ...   (periodic with period 2n, so any
// filter size works, even one larger than the image).
__device__ __forceinline__ int reflect_idx(int i, int n)
{
    if ((unsigned)i < (unsigned)n) return i;
    const int p = 2 * n;
    i %= p;
    if (i < 0) i += p;
    return (i < n) ? i : (p - 1 - i);
}

// Compare-exchange: a <- min, b <- max.
__device__ __forceinline__ void cas(u32& a, u32& b)
{
    const u32 lo = min(a, b);
    b = max(a, b);
    a = lo;
}

// ------------------------------------------------------------------------------------------------
// Kernel 1: small windows (K = 3, 5, 7, 9), selection in registers
// ------------------------------------------------------------------------------------------------
//
// 1. Every K-tall column segment of the tile is sorted once (optimal sorting network, registers)
//    into shared memory; it is shared by the K output pixels whose windows contain it.
// 2. Each output pixel reads its K sorted columns and sorts the K rows. Sorting the rows of a
//    column-sorted matrix keeps the columns sorted, so the window is now sorted along both axes.
// 3. In such a matrix the element at (row i, col c) has (i+1)(c+1)-1 elements <= it and
//    (K-i)(K-c)-1 elements >= it, so it can only be the median if both counts are <= (K*K-1)/2.
//    The excluded elements split evenly into "below" and "above" (the criterion is symmetric),
//    so the median of the whole window is the median of the remaining candidates (13 of 25 for
//    K=5, 29 of 49 for K=7, 47 of 81 for K=9), found by forgetful selection: from a working set of
//    (N+1)/2+1 values, the min and max provably cannot be the median, so they are dropped and the
//    next value is pulled in until three remain.
// For K=3 this reduces to the classic 19-compare-exchange network.

// Optimal sorting networks (Knuth, TAOCP 5.3.4): 3, 9, 16 and 25 compare-exchanges.
template <int N> __device__ __forceinline__ void sort_net(u32* v);
template <> __device__ __forceinline__ void sort_net<3>(u32* v)
{
    cas(v[0], v[1]); cas(v[1], v[2]); cas(v[0], v[1]);
}
template <> __device__ __forceinline__ void sort_net<5>(u32* v)
{
    cas(v[0], v[1]); cas(v[3], v[4]); cas(v[2], v[4]); cas(v[2], v[3]); cas(v[1], v[4]);
    cas(v[0], v[3]); cas(v[0], v[2]); cas(v[1], v[3]); cas(v[1], v[2]);
}
template <> __device__ __forceinline__ void sort_net<7>(u32* v)
{
    cas(v[1], v[2]); cas(v[3], v[4]); cas(v[5], v[6]); cas(v[0], v[2]); cas(v[3], v[5]); cas(v[4], v[6]);
    cas(v[0], v[1]); cas(v[4], v[5]); cas(v[2], v[6]); cas(v[0], v[4]); cas(v[1], v[5]); cas(v[0], v[3]);
    cas(v[2], v[5]); cas(v[1], v[3]); cas(v[2], v[4]); cas(v[2], v[3]);
}
template <> __device__ __forceinline__ void sort_net<9>(u32* v)
{
    cas(v[0], v[1]); cas(v[3], v[4]); cas(v[6], v[7]); cas(v[1], v[2]); cas(v[4], v[5]); cas(v[7], v[8]);
    cas(v[0], v[1]); cas(v[3], v[4]); cas(v[6], v[7]); cas(v[0], v[3]); cas(v[3], v[6]); cas(v[0], v[3]);
    cas(v[1], v[4]); cas(v[4], v[7]); cas(v[1], v[4]); cas(v[2], v[5]); cas(v[5], v[8]); cas(v[2], v[5]);
    cas(v[1], v[3]); cas(v[5], v[7]); cas(v[2], v[6]); cas(v[4], v[6]); cas(v[2], v[4]); cas(v[2], v[3]);
    cas(v[5], v[6]);
}

// Median candidates of a row- and column-sorted K x K matrix (see above), in raster order.
__host__ __device__ constexpr bool is_candidate(int K, int i, int c)
{
    return (i + 1) * (c + 1) <= (K * K + 1) / 2 && (K - i) * (K - c) <= (K * K + 1) / 2;
}
__host__ __device__ constexpr int candidate_index(int K, int i, int c)   // = number of candidates before (i, c)
{
    int n = 0;
    for (int ii = 0; ii < K; ++ii)
        for (int cc = 0; cc < K; ++cc)
            if ((ii < i || (ii == i && cc < c)) && is_candidate(K, ii, cc)) ++n;
    return n;
}
__host__ __device__ constexpr int num_candidates(int K) { return candidate_index(K, K, 0); }

// Median of N (odd) values by forgetful selection; all indexing is compile-time after unrolling,
// so everything stays in registers.
template <int N>
__device__ __forceinline__ u32 forgetful_median(const u32* e)
{
    constexpr int S = (N + 1) / 2 + 1;   // working set size; its min/max cannot be the median of N
    u32 w[S];
#pragma unroll
    for (int i = 0; i < S; ++i) w[i] = e[i];
#pragma unroll
    for (int i = S; i < N; ++i) {
        const int cur = 2 * S - i;       // current working-set size
#pragma unroll
        for (int j = 0; j < cur - 1; ++j) cas(w[j], w[j + 1]);   // max -> w[cur-1]
#pragma unroll
        for (int j = cur - 2; j > 0; --j) cas(w[j - 1], w[j]);   // min -> w[0]
        w[0] = e[i];                                             // drop min (overwrite) and max (ignore w[cur-1])
    }
    return max(min(w[0], w[1]), min(max(w[0], w[1]), w[2]));    // survivors have ranks m-1, m, m+1
}

template <int K>
__global__ void __launch_bounds__(256)
medfilt_small_kernel(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    constexpr int R = K / 2, TW = 32, TH = 8, SW = TW + K - 1;
    __shared__ u32 scol[TH * K * SW];    // sorted column segments: element i of (lx, ly) at scol[(ly*K + i)*SW + lx]

    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;
    const int tid = threadIdx.y * TW + threadIdx.x;

    // Phase 1: sort every K-tall column segment of the tile.
    for (int p = tid; p < SW * TH; p += TW * TH) {
        const int ly = p / SW, lx = p - ly * SW;
        const int gx = reflect_idx(x0 + lx - R, W);
        const int gy0 = y0 + ly - R;
        u32 v[K];
#pragma unroll
        for (int i = 0; i < K; ++i) v[i] = f2k(__ldg(in + (size_t)reflect_idx(gy0 + i, H) * W + gx));
        sort_net<K>(v);
#pragma unroll
        for (int i = 0; i < K; ++i) scol[(ly * K + i) * SW + lx] = v[i];
    }
    __syncthreads();

    const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= W || y >= H) return;

    // Phase 2: median of the window from its K sorted columns.
    const u32* base = scol + (threadIdx.y * K) * SW + threadIdx.x;   // i-th smallest of column c: base[i*SW + c]
    u32 med;

    if constexpr (K == 3) {
        const u32 lo  = max(max(base[0], base[1]), base[2]);                       // largest column minimum
        const u32 hi  = min(min(base[2 * SW], base[2 * SW + 1]), base[2 * SW + 2]); // smallest column maximum
        const u32 m0 = base[SW], m1 = base[SW + 1], m2 = base[SW + 2];
        const u32 mid = max(min(m0, m1), min(max(m0, m1), m2));                    // median of column medians
        med = max(min(lo, mid), min(max(lo, mid), hi));
    } else {
        u32 a[K][K];                     // a[i][c]: i-th smallest of column c
#pragma unroll
        for (int i = 0; i < K; ++i)
#pragma unroll
            for (int c = 0; c < K; ++c) a[i][c] = base[i * SW + c];
#pragma unroll
        for (int i = 0; i < K; ++i) sort_net<K>(a[i]);   // rows sorted, columns remain sorted

        constexpr int NC = num_candidates(K);
        u32 cand[NC];
#pragma unroll
        for (int i = 0; i < K; ++i)
#pragma unroll
            for (int c = 0; c < K; ++c)
                if (is_candidate(K, i, c)) cand[candidate_index(K, i, c)] = a[i][c];
        med = forgetful_median<NC>(cand);
    }

    out[(size_t)y * W + x] = k2f(med);
}

// ------------------------------------------------------------------------------------------------
// Kernel 2: incrementally maintained sorted columns in shared memory + snapping bisection
// ------------------------------------------------------------------------------------------------
//
// A block of TW threads owns a TW-wide strip and walks it down `rows` output rows. Shared memory
// holds one sorted K-tall column segment for each of the TW+K-1 columns the strip's windows touch
// (scol[i*SW + c] = i-th smallest of column c). The columns are sorted once for the first row and
// then, for every following row, updated in O(K): the value leaving the window is removed and the
// value entering it is inserted in place.
//
// For each output pixel the median is found by bisection on the key value: the number of window
// elements <= probe is a binary search in each of its K sorted columns. After every probe the
// bracket [lo, hi] is snapped to actual window elements (largest <= probe / smallest > probe), so
// the search needs about log2(#distinct values in the window) probes rather than 32.

// Count the elements <= probe in ILP adjacent sorted columns (cols[i*SW + j], i < K, j < ILP) and
// track the largest element <= probe / smallest element > probe. All columns take the same
// ceil(log2 K) binary-search steps, so they are searched in lockstep: ILP independent
// shared-memory loads per step instead of one dependent chain, which hides the latency at the low
// occupancy that large K forces.
template <int ILP>
__device__ __forceinline__ void bisect_columns(const u32* __restrict__ cols, int K, int SW, u32 probe,
                                               int& cnt, u32& maxLE, u32& minGT)
{
    int b[ILP];
#pragma unroll
    for (int j = 0; j < ILP; ++j) b[j] = 0;
    for (int n = K; n > 1; n -= n >> 1) {            // branchless upper_bound: idx in [b, b+n]
        const int half = n >> 1;
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            b[j] = (cols[(b[j] + half) * SW + j] <= probe) ? b[j] + half : b[j];
    }
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
        const u32* col = cols + j;
        const int idx = b[j] + ((col[b[j] * SW] <= probe) ? 1 : 0);   // number of elements <= probe
        cnt += idx;
        const u32 le = col[max(idx - 1, 0) * SW];
        const u32 gt = col[min(idx, K - 1) * SW];
        maxLE = (idx > 0) ? max(maxLE, le) : maxLE;
        minGT = (idx < K) ? min(minGT, gt) : minGT;
    }
}

// Sorted column (element i at col[i*SW]): replace one occurrence of `a` (guaranteed present) by `b`,
// keeping it sorted. Cost O(|rank(b) - rank(a)|).
__device__ __forceinline__ void column_replace(u32* col, int K, int SW, u32 a, u32 b)
{
    int p = 0, n = K;                                // branchless lower_bound: first index with col >= a
    while (n > 1) {
        const int half = n >> 1;
        p = (col[(p + half) * SW] < a) ? p + half : p;
        n -= half;
    }
    p += (col[p * SW] < a) ? 1 : 0;
    if (b >= a) {
        while (p < K - 1) {
            const u32 nx = col[(p + 1) * SW];
            if (nx > b) break;
            col[p * SW] = nx;
            ++p;
        }
    } else {
        while (p > 0) {
            const u32 pv = col[(p - 1) * SW];
            if (pv <= b) break;
            col[p * SW] = pv;
            --p;
        }
    }
    col[p * SW] = b;
}

// Block = TW threads; SW = TW+K-1 rounded up to a multiple of 32 so that a warp reading 32
// consecutive columns at *different* row indices is still bank-conflict free.
template <int TW>
__global__ void __launch_bounds__(TW)
medfilt_bisect_kernel(const float* __restrict__ in, float* __restrict__ out, int W, int H, int K, int SW, int rows)
{
    extern __shared__ u32 scol[];

    const int R = K / 2;
    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * rows;
    const int tx = threadIdx.x;
    const int ncols = TW + K - 1;
    const int x = x0 + tx;
    const int target = (K * K + 1) / 2;              // median = smallest v with count(<= v) >= target

    // Sorted columns for the first row (insertion sort straight into shared memory).
    for (int c = tx; c < ncols; c += TW) {
        const int gx = reflect_idx(x0 + c - R, W);
        u32* col = scol + c;
        for (int i = 0; i < K; ++i) {
            const u32 v = f2k(__ldg(in + (size_t)reflect_idx(y0 - R + i, H) * W + gx));
            int j = i;
            while (j > 0) {
                const u32 prev = col[(j - 1) * SW];
                if (prev <= v) break;
                col[j * SW] = prev;
                --j;
            }
            col[j * SW] = v;
        }
    }
    __syncthreads();

    for (int r = 0; r < rows; ++r) {
        const int y = y0 + r;
        if (r > 0) {
            // Slide every column down one row: drop row y-1-R, insert row y+R.
            const float* row_out = in + (size_t)reflect_idx(y - 1 - R, H) * W;
            const float* row_in  = in + (size_t)reflect_idx(y + R, H) * W;
            for (int c = tx; c < ncols; c += TW) {
                const int gx = reflect_idx(x0 + c - R, W);
                column_replace(scol + c, K, SW, f2k(__ldg(row_out + gx)), f2k(__ldg(row_in + gx)));
            }
            __syncthreads();
        }

        if (x < W && y < H) {
            const u32* base = scol + tx;              // column c, element i: base[i*SW + c]

            // Initial bracket: [smallest column median, largest column median]. Fewer than
            // K*(K-1)/2 <= (K*K-1)/2 window elements are below the smallest column median (at most
            // (K-1)/2 per column), so the window median is >= it; symmetrically for the largest.
            u32 lo = 0xFFFFFFFFu, hi = 0u;
            for (int c = 0; c < K; ++c) {
                const u32 m = base[R * SW + c];
                lo = min(lo, m);
                hi = max(hi, m);
            }

            while (lo < hi) {
                const u32 probe = lo + ((hi - lo) >> 1);
                int cnt = 0;
                u32 maxLE = 0u, minGT = 0xFFFFFFFFu;
                int c = 0;
                for (; c + 8 <= K; c += 8) bisect_columns<8>(base + c, K, SW, probe, cnt, maxLE, minGT);
                if (c + 4 <= K) { bisect_columns<4>(base + c, K, SW, probe, cnt, maxLE, minGT); c += 4; }
                if (c + 2 <= K) { bisect_columns<2>(base + c, K, SW, probe, cnt, maxLE, minGT); c += 2; }
                if (c < K)        bisect_columns<1>(base + c, K, SW, probe, cnt, maxLE, minGT);
                // Snap the bracket to actual window elements: the median is one of them.
                if (cnt >= target) hi = maxLE; else lo = minGT;
            }
            out[(size_t)y * W + x] = k2f(lo);
        }
        __syncthreads();                             // everyone is done reading before the next update
    }
}

// ------------------------------------------------------------------------------------------------
// Kernel 3: huge windows -- sorted columns in global scratch memory, sample table in shared memory
// ------------------------------------------------------------------------------------------------
//
// Same scheme as kernel 2 (incrementally maintained sorted columns, snapping bisection), for windows
// whose sorted columns no longer fit in shared memory. Persistent blocks of TW threads loop over
// TW x rows strips; each block owns a private scratch slot holding the sorted column segments
// column-contiguously (column c, element i at cols[c*KP + i], KP = K rounded up to G, padded with
// 0xFFFFFFFF). Shared memory holds every G-th element of every column (samp[j*SWs + c] =
// cols[c*KP + j*G]) plus the column medians. A probe's count(<= probe) per column is then a binary
// search over the samples in shared memory followed by a single G-element (one or a few 32-byte
// sectors) read from global memory, instead of log2(K) dependent global loads.

// Merge sort of a[0..K) using tmp[0..K) as scratch (one thread, sequential access).
__device__ void column_sort(u32* a, u32* tmp, int K)
{
    u32* src = a;
    u32* dst = tmp;
    for (int w = 1; w < K; w *= 2) {
        for (int lo = 0; lo < K; lo += 2 * w) {
            const int mid = min(lo + w, K), hi = min(lo + 2 * w, K);
            int i = lo, j = mid, o = lo;
            while (i < mid && j < hi) {
                const u32 x = src[i], y = src[j];
                if (y < x) { dst[o++] = y; ++j; } else { dst[o++] = x; ++i; }
            }
            while (i < mid) dst[o++] = src[i++];
            while (j < hi)  dst[o++] = src[j++];
        }
        u32* t = src; src = dst; dst = t;
    }
    if (src != a)
        for (int i = 0; i < K; ++i) a[i] = src[i];
}

// Sorted column (stride `stride`): replace one occurrence of `a` (guaranteed present) by `b`, keeping
// it sorted; returns the range of indices that changed via lo_i/hi_i.
__device__ __forceinline__ void column_replace_range(u32* col, int K, int stride, u32 a, u32 b, int& lo_i, int& hi_i)
{
    int p = 0, n = K;                                // branchless lower_bound: first index with col >= a
    while (n > 1) {
        const int half = n >> 1;
        p = (col[(p + half) * stride] < a) ? p + half : p;
        n -= half;
    }
    p += (col[p * stride] < a) ? 1 : 0;
    const int pa = p;
    if (b >= a) {
        while (p < K - 1) {
            const u32 nx = col[(p + 1) * stride];
            if (nx > b) break;
            col[p * stride] = nx;
            ++p;
        }
    } else {
        while (p > 0) {
            const u32 pv = col[(p - 1) * stride];
            if (pv <= b) break;
            col[p * stride] = pv;
            --p;
        }
    }
    col[p * stride] = b;
    lo_i = min(pa, p);
    hi_i = max(pa, p);
}

// Load one G-element group of a sorted column and count its elements <= probe; `le` is the largest
// element <= probe (valid if cntg > 0), `gt` the smallest element > probe (valid if cntg < G).
template <int G>
__device__ __forceinline__ void huge_group(const u32* grp, u32 probe, int& cntg, u32& le, u32& gt)
{
    u32 g[G];
#pragma unroll
    for (int q = 0; q < G / 4; ++q) {
        const uint4 v = reinterpret_cast<const uint4*>(grp)[q];   // plain load: scratch is written in this kernel
        g[4 * q] = v.x; g[4 * q + 1] = v.y; g[4 * q + 2] = v.z; g[4 * q + 3] = v.w;
    }
    cntg = 0; le = 0u; gt = 0xFFFFFFFFu;
#pragma unroll
    for (int i = 0; i < G; ++i) {
        const bool b = g[i] <= probe;
        cntg += b ? 1 : 0;
        le = b ? g[i] : le;
    }
#pragma unroll
    for (int i = G - 1; i >= 0; --i) gt = (g[i] > probe) ? g[i] : gt;
}

// ILP adjacent columns (samples at samp[j*SWs + i], data at cols + i*KP), counted in lockstep.
template <int G, int ILP>
__device__ __forceinline__ void huge_columns(const u32* samp, int nsamp, int SWs, const u32* cols, int KP, int K,
                                             u32 probe, int& cnt, u32& maxLE, u32& minGT)
{
    int b[ILP];
#pragma unroll
    for (int j = 0; j < ILP; ++j) b[j] = 0;
    for (int n = nsamp; n > 1; n -= n >> 1) {
        const int half = n >> 1;
#pragma unroll
        for (int j = 0; j < ILP; ++j)
            b[j] = (samp[(b[j] + half) * SWs + j] <= probe) ? b[j] + half : b[j];
    }
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
        const int jmax = b[j] + ((samp[b[j] * SWs + j] <= probe) ? 1 : 0) - 1;   // last sample <= probe, or -1
        if (jmax < 0) {                                                         // nothing <= probe: idx = 0
            minGT = min(minGT, samp[j]);
            continue;
        }
        int cntg;
        u32 le, gt;
        huge_group<G>(cols + (size_t)j * KP + jmax * G, probe, cntg, le, gt);   // cntg >= 1: group starts with a sample <= probe
        const int idx = jmax * G + cntg;
        cnt += idx;
        maxLE = max(maxLE, le);
        if (idx < K) minGT = min(minGT, (cntg < G) ? gt : samp[(jmax + 1) * SWs + j]);
    }
}

template <int TW, int G>
__global__ void __launch_bounds__(TW)
medfilt_huge_kernel(const float* __restrict__ in, float* __restrict__ out, int W, int H, int K, int KP,
                    int SWs, int rows, int nstripx, int nstrips, u32* scratch)
{
    extern __shared__ u32 samp[];                    // [nsamp][SWs] samples, then [SWs] column medians

    const int R = K / 2, tx = threadIdx.x;
    const int ncols = TW + K - 1;
    const int nsamp = (K + G - 1) / G;
    u32* medrow = samp + (size_t)nsamp * SWs;
    u32* cols = scratch + (size_t)blockIdx.x * ncols * KP * 2;
    u32* tmps = cols + (size_t)ncols * KP;
    const int target = (K * K + 1) / 2;

    for (int strip = blockIdx.x; strip < nstrips; strip += gridDim.x) {
        const int x0 = (strip % nstripx) * TW, y0 = (strip / nstripx) * rows;
        const int x = x0 + tx;

        // Sorted columns + samples for the strip's first row.
        for (int c = tx; c < ncols; c += TW) {
            const int gx = reflect_idx(x0 + c - R, W);
            u32* col = cols + (size_t)c * KP;
            for (int i = 0; i < K; ++i) col[i] = f2k(__ldg(in + (size_t)reflect_idx(y0 - R + i, H) * W + gx));
            for (int i = K; i < KP; ++i) col[i] = 0xFFFFFFFFu;
            column_sort(col, tmps + (size_t)c * KP, K);
            for (int j = 0; j < nsamp; ++j) samp[j * SWs + c] = col[j * G];
            medrow[c] = col[R];
        }
        __syncthreads();

        for (int r = 0; r < rows; ++r) {
            const int y = y0 + r;
            if (y >= H) break;                       // uniform across the block
            if (r > 0) {
                const float* row_out = in + (size_t)reflect_idx(y - 1 - R, H) * W;
                const float* row_in  = in + (size_t)reflect_idx(y + R, H) * W;
                for (int c = tx; c < ncols; c += TW) {
                    const int gx = reflect_idx(x0 + c - R, W);
                    u32* col = cols + (size_t)c * KP;
                    int lo_i, hi_i;
                    column_replace_range(col, K, 1, f2k(__ldg(row_out + gx)), f2k(__ldg(row_in + gx)), lo_i, hi_i);
                    for (int j = (lo_i + G - 1) / G; j * G <= hi_i; ++j) samp[j * SWs + c] = col[j * G];
                    if (lo_i <= R && R <= hi_i) medrow[c] = col[R];
                }
                __syncthreads();
            }

            if (x < W) {
                u32 lo = 0xFFFFFFFFu, hi = 0u;       // bracket: [smallest, largest] column median
                for (int c = 0; c < K; ++c) {
                    const u32 m = medrow[tx + c];
                    lo = min(lo, m);
                    hi = max(hi, m);
                }
                while (lo < hi) {
                    const u32 probe = lo + ((hi - lo) >> 1);
                    int cnt = 0;
                    u32 maxLE = 0u, minGT = 0xFFFFFFFFu;
                    int c = 0;
                    for (; c + 8 <= K; c += 8)
                        huge_columns<G, 8>(samp + tx + c, nsamp, SWs, cols + (size_t)(tx + c) * KP, KP, K, probe, cnt, maxLE, minGT);
                    if (c + 4 <= K) { huge_columns<G, 4>(samp + tx + c, nsamp, SWs, cols + (size_t)(tx + c) * KP, KP, K, probe, cnt, maxLE, minGT); c += 4; }
                    if (c + 2 <= K) { huge_columns<G, 2>(samp + tx + c, nsamp, SWs, cols + (size_t)(tx + c) * KP, KP, K, probe, cnt, maxLE, minGT); c += 2; }
                    if (c < K)        huge_columns<G, 1>(samp + tx + c, nsamp, SWs, cols + (size_t)(tx + c) * KP, KP, K, probe, cnt, maxLE, minGT);
                    if (cnt >= target) hi = maxLE; else lo = minGT;
                }
                out[(size_t)y * W + x] = k2f(lo);
            }
            __syncthreads();                         // reads done before the next update / strip
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Kernel 4: fallback for windows that do not fit anywhere (reads through L1/L2)
// ------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
medfilt_global_kernel(const float* __restrict__ in, float* __restrict__ out, int W, int H, int K)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int R = K / 2;
    const int target = (K * K + 1) / 2;

    u32 lo = 0xFFFFFFFFu, hi = 0u;
    for (int dy = -R; dy <= R; ++dy) {
        const float* row = in + (size_t)reflect_idx(y + dy, H) * W;
        for (int dx = -R; dx <= R; ++dx) {
            const u32 k = f2k(__ldg(row + reflect_idx(x + dx, W)));
            lo = min(lo, k);
            hi = max(hi, k);
        }
    }
    while (lo < hi) {
        const u32 probe = lo + ((hi - lo) >> 1);
        int cnt = 0;
        u32 maxLE = 0u, minGT = 0xFFFFFFFFu;
        for (int dy = -R; dy <= R; ++dy) {
            const float* row = in + (size_t)reflect_idx(y + dy, H) * W;
            for (int dx = -R; dx <= R; ++dx) {
                const u32 k = f2k(__ldg(row + reflect_idx(x + dx, W)));
                const bool le = k <= probe;
                cnt += le ? 1 : 0;
                maxLE = le ? max(maxLE, k) : maxLE;
                minGT = le ? minGT : min(minGT, k);
            }
        }
        if (cnt >= target) hi = maxLE; else lo = minGT;
    }
    out[(size_t)y * W + x] = k2f(lo);
}

// ------------------------------------------------------------------------------------------------
// Host-side launch helpers
// ------------------------------------------------------------------------------------------------

template <int K>
cudaError_t launch_small(const float* in, float* out, int W, int H, cudaStream_t s)
{
    const dim3 block(32, 8), grid((W + 31) / 32, (H + 7) / 8);
    medfilt_small_kernel<K><<<grid, block, 0, s>>>(in, out, W, H);
    return cudaGetLastError();
}

int device_attr(cudaDeviceAttr attr, int fallback)
{
    int dev = 0, v = fallback;
    if (cudaGetDevice(&dev) == cudaSuccess) cudaDeviceGetAttribute(&v, attr, dev);
    return v;
}

size_t shared_mem_optin() { return (size_t)device_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, 48 * 1024); }

template <int TW>
size_t bisect_smem_bytes(int K, int& SW)
{
    SW = (TW + K - 1 + 31) & ~31;
    return (size_t)K * SW * sizeof(u32);
}

// Rows per block: as many as possible (the first row's full column sort is amortised over them),
// but keep at least a couple of blocks per SM in flight.
int bisect_rows_per_block(int W, int H, int TW)
{
    const size_t min_blocks = 2 * (size_t)device_attr(cudaDevAttrMultiProcessorCount, 16);
    const size_t bx = (W + TW - 1) / TW;
    for (int rows = 16; rows > 1; rows >>= 1)
        if (bx * ((H + rows - 1) / rows) >= min_blocks) return rows;
    return 1;
}

template <int TW>
cudaError_t launch_bisect(const float* in, float* out, int W, int H, int K, cudaStream_t s)
{
    int SW;
    const size_t smem = bisect_smem_bytes<TW>(K, SW);
    if (smem > shared_mem_optin()) return cudaErrorInvalidValue;
    static bool configured = false;      // once per instantiation: opt in to the full shared memory
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(medfilt_bisect_kernel<TW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             (int)shared_mem_optin());
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(medfilt_bisect_kernel<TW>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
        if (e != cudaSuccess) return e;
        configured = true;
    }
    const int rows = bisect_rows_per_block(W, H, TW);
    const dim3 block(TW), grid((W + TW - 1) / TW, (H + rows - 1) / rows);
    medfilt_bisect_kernel<TW><<<grid, block, smem, s>>>(in, out, W, H, K, SW, rows);
    return cudaGetLastError();
}

void ensure_mem_pool()
{
    // Keep freed device memory cached in the default pool so repeated calls do not pay for
    // cudaMalloc/cudaFree (and their implicit synchronisation) every time.
    static std::once_flag once;
    std::call_once(once, [] {
        int dev = 0;
        cudaMemPool_t pool;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
            unsigned long long threshold = ~0ull;   // cuuint64_t
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
    });
}

// Sample group size G is the smallest (8, 16, 32, 64) whose sample table fits in shared memory.
// `dry` only checks feasibility.
template <int TW, int G>
cudaError_t launch_huge(const float* in, float* out, int W, int H, int K, cudaStream_t s, bool dry)
{
    const int ncols = TW + K - 1;
    const int SWs = (ncols + 31) & ~31;
    const int nsamp = (K + G - 1) / G;
    const size_t smem = (size_t)(nsamp + 1) * SWs * sizeof(u32);
    if (smem > shared_mem_optin()) return cudaErrorInvalidValue;
    if (dry) return cudaSuccess;

    static bool configured = false;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(medfilt_huge_kernel<TW, G>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             (int)shared_mem_optin());
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(medfilt_huge_kernel<TW, G>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
        if (e != cudaSuccess) return e;
        configured = true;
    }

    // Persistent grid: as many blocks as fit on the device at once, each owning one scratch slot.
    int per_sm = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, medfilt_huge_kernel<TW, G>, TW, smem);
    if (e != cudaSuccess) return e;
    if (per_sm < 1) return cudaErrorInvalidValue;
    const int resident = per_sm * device_attr(cudaDevAttrMultiProcessorCount, 16);
    const int nstripx = (W + TW - 1) / TW;
    int rows = 64;                                   // the first row's full column sort is amortised over these
    while (rows > 1 && (size_t)nstripx * ((H + rows - 1) / rows) < (size_t)2 * resident) rows >>= 1;
    const int nstrips = nstripx * ((H + rows - 1) / rows);
    const int grid = nstrips < resident ? nstrips : resident;
    const int KP = nsamp * G;

    ensure_mem_pool();
    u32* scratch = nullptr;
    const size_t scratch_bytes = (size_t)grid * ncols * KP * 2 * sizeof(u32);   // columns + merge-sort temp
    if ((e = cudaMallocAsync((void**)&scratch, scratch_bytes, s)) != cudaSuccess) return e;
    medfilt_huge_kernel<TW, G><<<grid, TW, smem, s>>>(in, out, W, H, K, KP, SWs, rows, nstripx, nstrips, scratch);
    e = cudaGetLastError();
    cudaFreeAsync(scratch, s);                       // stream-ordered: released after the kernel
    return e;
}

cudaError_t launch_huge_auto(const float* in, float* out, int W, int H, int K, cudaStream_t s, bool dry)
{
    cudaError_t e;
    if ((e = launch_huge<128, 8>(in, out, W, H, K, s, dry))  != cudaErrorInvalidValue) return e;
    if ((e = launch_huge<128, 16>(in, out, W, H, K, s, dry)) != cudaErrorInvalidValue) return e;
    if ((e = launch_huge<128, 32>(in, out, W, H, K, s, dry)) != cudaErrorInvalidValue) return e;
    return launch_huge<128, 64>(in, out, W, H, K, s, dry);
}

cudaError_t launch_global(const float* in, float* out, int W, int H, int K, cudaStream_t s)
{
    const dim3 block(32, 8), grid((W + 31) / 32, (H + 7) / 8);
    medfilt_global_kernel<<<grid, block, 0, s>>>(in, out, W, H, K);
    return cudaGetLastError();
}

bool bisect_fits(int algo, int K)
{
    int SW;
    const size_t lim = shared_mem_optin();
    switch (algo) {
        case GPUMEDFILT_ALGO_BISECT_128: return bisect_smem_bytes<128>(K, SW) <= lim;
        case GPUMEDFILT_ALGO_BISECT_256: return bisect_smem_bytes<256>(K, SW) <= lim;
        case GPUMEDFILT_ALGO_HUGE:       return launch_huge_auto(nullptr, nullptr, 1, 1, K, 0, true) == cudaSuccess;
        default: return false;
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Internal device-pointer API
// ------------------------------------------------------------------------------------------------

const char* gpumedfilt_algo_name(int algo)
{
    switch (algo) {
        case GPUMEDFILT_ALGO_AUTO:       return "auto";
        case GPUMEDFILT_ALGO_SMALL:      return "small(regs)";
        case GPUMEDFILT_ALGO_BISECT_128: return "bisect128";
        case GPUMEDFILT_ALGO_BISECT_256: return "bisect256";
        case GPUMEDFILT_ALGO_HUGE:       return "huge";
        case GPUMEDFILT_ALGO_GLOBAL:     return "global";
        default:                         return "?";
    }
}

int gpumedfilt_auto_algo(int K)
{
    if (K == 3 || K == 5 || K == 7 || K == 9) return GPUMEDFILT_ALGO_SMALL;
    if (bisect_fits(GPUMEDFILT_ALGO_BISECT_128, K)) return GPUMEDFILT_ALGO_BISECT_128;
    if (bisect_fits(GPUMEDFILT_ALGO_HUGE, K)) return GPUMEDFILT_ALGO_HUGE;
    return GPUMEDFILT_ALGO_GLOBAL;
}

cudaError_t gpumedfilt_device(const float* d_in, float* d_out, int W, int H, int K, cudaStream_t s, int algo)
{
    if (!d_in || !d_out || W <= 0 || H <= 0 || K < 1 || (K & 1) == 0) return cudaErrorInvalidValue;
    if (K == 1) return cudaMemcpyAsync(d_out, d_in, (size_t)W * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    if (algo == GPUMEDFILT_ALGO_AUTO) algo = gpumedfilt_auto_algo(K);

    switch (algo) {
        case GPUMEDFILT_ALGO_SMALL:
            switch (K) {
                case 3:  return launch_small<3>(d_in, d_out, W, H, s);
                case 5:  return launch_small<5>(d_in, d_out, W, H, s);
                case 7:  return launch_small<7>(d_in, d_out, W, H, s);
                case 9:  return launch_small<9>(d_in, d_out, W, H, s);
                default: return cudaErrorInvalidValue;
            }
        case GPUMEDFILT_ALGO_BISECT_128: return launch_bisect<128>(d_in, d_out, W, H, K, s);
        case GPUMEDFILT_ALGO_BISECT_256: return launch_bisect<256>(d_in, d_out, W, H, K, s);
        case GPUMEDFILT_ALGO_HUGE:       return launch_huge_auto(d_in, d_out, W, H, K, s, false);
        case GPUMEDFILT_ALGO_GLOBAL:     return launch_global(d_in, d_out, W, H, K, s);
        default:                         return cudaErrorInvalidValue;
    }
}

// ------------------------------------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------------------------------------

extern "C" GPUMEDFILT_API void gpu_medfilt2(const float* input, float* output, int width, int height, int filter_size)
{
    if (!input || !output || width <= 0 || height <= 0) {
        fprintf(stderr, "gpu_medfilt2: invalid arguments (input=%p output=%p width=%d height=%d)\n",
                (const void*)input, (const void*)output, width, height);
        return;
    }
    if (filter_size < 1 || (filter_size & 1) == 0) {
        fprintf(stderr, "gpu_medfilt2: filter_size must be a positive odd number (got %d)\n", filter_size);
        return;
    }
    const size_t bytes = (size_t)width * height * sizeof(float);
    if (filter_size == 1) {   // median of a single element
        memcpy(output, input, bytes);
        return;
    }

    ensure_mem_pool();

    const cudaStream_t s = cudaStreamPerThread;
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaError_t e = cudaSuccess;
    const char* stage = "";

    do {
        stage = "cudaMallocAsync";
        if ((e = cudaMallocAsync((void**)&d_in, bytes, s)) != cudaSuccess) break;
        if ((e = cudaMallocAsync((void**)&d_out, bytes, s)) != cudaSuccess) break;
        stage = "host->device copy";
        if ((e = cudaMemcpyAsync(d_in, input, bytes, cudaMemcpyHostToDevice, s)) != cudaSuccess) break;
        stage = "kernel launch";
        if ((e = gpumedfilt_device(d_in, d_out, width, height, filter_size, s, GPUMEDFILT_ALGO_AUTO)) != cudaSuccess) break;
        stage = "device->host copy";
        if ((e = cudaMemcpyAsync(output, d_out, bytes, cudaMemcpyDeviceToHost, s)) != cudaSuccess) break;
    } while (0);

    if (d_in)  cudaFreeAsync(d_in, s);
    if (d_out) cudaFreeAsync(d_out, s);
    const cudaError_t sync = cudaStreamSynchronize(s);
    if (e == cudaSuccess && sync != cudaSuccess) { e = sync; stage = "synchronize"; }
    if (e != cudaSuccess)
        fprintf(stderr, "gpu_medfilt2: %s failed: %s\n", stage, cudaGetErrorString(e));
}
