// Correctness tests (bit-exact against a CPU reference) and benchmarks for gpu_medfilt2.
//
//   medfilt_test --test     correctness only
//   medfilt_test --bench    benchmarks only
//   medfilt_test            both

#include "gpumedfilt.h"
#include "gpumedfilt_internal.h"

#include <cuda_runtime.h>
#ifdef GPUMEDFILT_HAVE_NPP
#include <nppi_filtering_functions.h>
#include <nppdefs.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t _e = (x);                                                                      \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e));  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// ---- CPU reference (same total order on float bit patterns as the GPU) --------------------------

static uint32_t f2k(float f) { uint32_t u; memcpy(&u, &f, 4); return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u); }
static float    k2f(uint32_t k) { uint32_t u = k ^ ((k >> 31) ? 0x80000000u : 0xFFFFFFFFu); float f; memcpy(&f, &u, 4); return f; }

static int reflect(int i, int n)
{
    if (i >= 0 && i < n) return i;
    const int p = 2 * n;
    i %= p;
    if (i < 0) i += p;
    return i < n ? i : p - 1 - i;
}

static void cpu_medfilt2(const float* in, float* out, int W, int H, int K)
{
    const int R = K / 2;
    std::vector<uint32_t> buf((size_t)K * K);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t n = 0;
            for (int dy = -R; dy <= R; ++dy) {
                const float* row = in + (size_t)reflect(y + dy, H) * W;
                for (int dx = -R; dx <= R; ++dx) buf[n++] = f2k(row[reflect(x + dx, W)]);
            }
            std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
            out[(size_t)y * W + x] = k2f(buf[n / 2]);
        }
}

// ---- test images --------------------------------------------------------------------------------

static std::vector<float> make_image(int W, int H, unsigned seed, bool specials)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 9);
    const float special[] = { 0.0f, -0.0f, 1.0f, 0.5f, 1e-40f, INFINITY, -INFINITY, NAN, 0.25f, -0.25f };
    std::vector<float> img((size_t)W * H);
    for (auto& v : img) {
        v = uni(rng);
        if (specials && pick(rng) < 3) v = special[pick(rng)];   // 30% duplicates / special values
    }
    return img;
}

static size_t count_mismatch(const std::vector<float>& a, const std::vector<float>& b, size_t* first)
{
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], 4) != 0) { if (!bad) *first = i; ++bad; }
    return bad;
}

// ---- correctness --------------------------------------------------------------------------------

static bool run_tests()
{
    struct Case { int W, H; std::vector<int> Ks; };
    const std::vector<Case> cases = {
        { 1,   1,   { 1, 3, 5, 9 } },
        { 1,   9,   { 3, 7, 21 } },
        { 9,   1,   { 3, 7, 21 } },
        { 5,   3,   { 3, 5, 11, 31 } },
        { 33,  17,  { 3, 5, 7, 9, 11, 15, 21, 41 } },
        { 100, 64,  { 3, 5, 7, 9, 13, 17, 25, 63 } },
        { 257, 131, { 3, 5, 7, 9, 11, 15, 21, 31, 101 } },
        { 64,  40,  { 121, 151, 255 } },
        { 200, 90,  { 201 } },
    };

    int failures = 0, checks = 0;
    for (const Case& c : cases) {
        for (int K : c.Ks) {
            const std::vector<float> in = make_image(c.W, c.H, 1234u + c.W * 7 + K, true);
            std::vector<float> ref((size_t)c.W * c.H), got((size_t)c.W * c.H, -12345.0f);
            cpu_medfilt2(in.data(), ref.data(), c.W, c.H, K);

            // public host API (auto dispatch)
            gpu_medfilt2(in.data(), got.data(), c.W, c.H, K);
            size_t first = 0, bad = count_mismatch(ref, got, &first);
            ++checks;
            printf("  %4dx%-4d K=%-3d auto[%-11s] %s", c.W, c.H, K, gpumedfilt_algo_name(gpumedfilt_auto_algo(K)),
                   bad ? "FAIL" : "ok  ");
            if (bad) { ++failures; printf(" (%zu mismatches, first at %zu: got %g want %g)", bad, first, got[first], ref[first]); }
            printf("\n");

            // every kernel that can handle this K, through the device API
            if (K == 1) continue;
            float *d_in = nullptr, *d_out = nullptr;
            const size_t bytes = (size_t)c.W * c.H * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_in, bytes));
            CUDA_CHECK(cudaMalloc(&d_out, bytes));
            CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
            for (int algo = GPUMEDFILT_ALGO_SMALL; algo < GPUMEDFILT_ALGO_COUNT; ++algo) {
                CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
                const cudaError_t e = gpumedfilt_device(d_in, d_out, c.W, c.H, K, 0, algo);
                if (e == cudaErrorInvalidValue) continue;   // kernel not applicable to this K
                CUDA_CHECK(e);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
                bad = count_mismatch(ref, got, &first);
                ++checks;
                if (bad) {
                    ++failures;
                    printf("  %4dx%-4d K=%-3d      %-13s FAIL (%zu mismatches, first at %zu: got %g want %g)\n",
                           c.W, c.H, K, gpumedfilt_algo_name(algo), bad, first, got[first], ref[first]);
                } else {
                    printf("  %4dx%-4d K=%-3d      %-13s ok\n", c.W, c.H, K, gpumedfilt_algo_name(algo));
                }
            }
            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));
        }
    }
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0;
}

// ---- benchmarks ---------------------------------------------------------------------------------

#ifdef GPUMEDFILT_HAVE_NPP
static bool npp_median(const float* d_in, float* d_out, int W, int H, int K, NppiBorderType border,
                       cudaStream_t stream, Npp8u*& scratch, size_t& scratch_size)
{
    NppStreamContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.hStream = stream;
    CUDA_CHECK(cudaGetDevice(&ctx.nCudaDeviceId));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, ctx.nCudaDeviceId));
    ctx.nMultiProcessorCount = p.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = p.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = p.minor;

    const NppiSize size = { W, H }, mask = { K, K };
    const NppiPoint anchor = { K / 2, K / 2 }, offset = { 0, 0 };
    Npp32u need = 0;
    if (nppiFilterMedianBorderGetBufferSize_32f_C1R_Ctx(size, mask, &need, border, ctx) != NPP_SUCCESS) return false;
    if (need > scratch_size) {
        if (scratch) CUDA_CHECK(cudaFree(scratch));
        CUDA_CHECK(cudaMalloc(&scratch, need));
        scratch_size = need;
    }
    const NppStatus st = nppiFilterMedianBorder_32f_C1R_Ctx(d_in, W * (int)sizeof(float), size, offset, d_out,
                                                            W * (int)sizeof(float), size, mask, anchor, scratch, border, ctx);
    return st == NPP_SUCCESS;
}
#endif

// Best of `repeats` batches of `iters` launches (clocks ramp during the run; the minimum is the stable number).
template <class F>
static float time_ms(F&& launch, int iters, cudaStream_t stream, int repeats = 3)
{
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) launch();       // warm-up
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float best = 1e30f;
    for (int r = 0; r < repeats; ++r) {
        CUDA_CHECK(cudaEventRecord(a, stream));
        for (int i = 0; i < iters; ++i) launch();
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms / iters);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return best;
}

#ifdef GPUMEDFILT_HAVE_NPP
// NPP is run in its own process per K (medfilt_test --npp K): nppiFilterMedianBorder_32f_C1R hits an
// illegal memory access for some mask sizes, which poisons the CUDA context for everything after it.
static void run_npp(int K, int W, int H)
{
    cudaStream_t stream = 0;
    Npp8u* scratch = nullptr;
    size_t scratch_size = 0;

    // Do NPP's border modes match the required symmetric reflection? (small image, CPU reference)
    {
        const int w = 100, h = 64;
        const std::vector<float> in = make_image(w, h, 7u, false);
        std::vector<float> ref((size_t)w * h), got((size_t)w * h);
        cpu_medfilt2(in.data(), ref.data(), w, h, K);
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, ref.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_out, ref.size() * 4));
        CUDA_CHECK(cudaMemcpy(d_in, in.data(), ref.size() * 4, cudaMemcpyHostToDevice));
        for (NppiBorderType border : { NPP_BORDER_REPLICATE, NPP_BORDER_MIRROR }) {
            const char* name = border == NPP_BORDER_REPLICATE ? "replicate" : "mirror";
            if (!npp_median(d_in, d_out, w, h, K, border, stream, scratch, scratch_size)) {
                printf("K=%-3d NPP %-9s: unsupported\n", K, name);
                continue;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(got.data(), d_out, ref.size() * 4, cudaMemcpyDeviceToHost));
            size_t first = 0;
            const size_t bad = count_mismatch(ref, got, &first);
            printf("K=%-3d NPP %-9s: %zu of %zu pixels differ from the symmetric-border reference on %dx%d\n",
                   K, name, bad, ref.size(), w, h);
        }
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // Timing on the benchmark image.
    const size_t n = (size_t)W * H, bytes = n * sizeof(float);
    std::vector<float> img = make_image(W, H, 42u, false);
    std::vector<float> img8 = img;
    for (auto& v : img8) v = std::round(v * 127.0f) / 127.0f;
    float *d_in = nullptr, *d_in8 = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_in8, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, img.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in8, img8.data(), bytes, cudaMemcpyHostToDevice));
    for (NppiBorderType border : { NPP_BORDER_REPLICATE, NPP_BORDER_MIRROR }) {
        const char* name = border == NPP_BORDER_REPLICATE ? "NPP replicate" : "NPP mirror";
        if (!npp_median(d_in, d_out, W, H, K, border, stream, scratch, scratch_size)) continue;
        const float t8 = time_ms([&] { npp_median(d_in8, d_out, W, H, K, border, stream, scratch, scratch_size); }, 10, stream);
        const float tr = time_ms([&] { npp_median(d_in, d_out, W, H, K, border, stream, scratch, scratch_size); }, 10, stream);
        printf("%-4d | %-13s | %12.3f | %12.3f\n", K, name, t8, tr);
    }
    if (scratch) CUDA_CHECK(cudaFree(scratch));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_in8));
    CUDA_CHECK(cudaFree(d_out));
}
#endif

static void run_bench(int W, int H)
{
    const int Ks[] = { 3, 5, 7, 9, 11, 15, 21, 31, 63, 101, 201 };
    const size_t n = (size_t)W * H, bytes = n * sizeof(float);
    const double mpix = n / 1e6;
    printf("\nBenchmark %dx%d float image (%.1f MPix, %.1f MB). Kernel-only times are ms per image; "
           "host->host is the public gpu_medfilt2 call including transfers.\n\n", W, H, mpix, bytes / 1e6);

    // "8-bit-like" content (values quantised to 256 levels) is the common case for the bisection kernel;
    // continuous random floats are its worst case. Both are timed.
    std::vector<float> img = make_image(W, H, 42u, false);
    std::vector<float> img8 = img;
    for (auto& v : img8) v = std::round(v * 127.0f) / 127.0f;
    std::vector<float> out(n);

    float *d_in = nullptr, *d_in8 = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_in8, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, img.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in8, img8.data(), bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream = 0;

    printf("%-4s | %-13s | %12s | %12s\n", "K", "kernel", "8bit-like ms", "random ms");
    printf("-----+---------------+--------------+-------------\n");
    for (int K : Ks) {
        const int auto_algo = gpumedfilt_auto_algo(K);
        for (int algo = GPUMEDFILT_ALGO_SMALL; algo < GPUMEDFILT_ALGO_COUNT; ++algo) {
            if (algo == GPUMEDFILT_ALGO_GLOBAL && K > 15) continue;   // too slow to be interesting
            if (gpumedfilt_device(d_in, d_out, W, H, K, stream, algo) == cudaErrorInvalidValue) continue;
            CUDA_CHECK(cudaGetLastError());
            const float t8 = time_ms([&] { CUDA_CHECK(gpumedfilt_device(d_in8, d_out, W, H, K, stream, algo)); }, 10, stream);
            const float tr = time_ms([&] { CUDA_CHECK(gpumedfilt_device(d_in, d_out, W, H, K, stream, algo)); }, 10, stream);
            printf("%-4d | %-13s | %12.3f | %12.3f%s\n", K, gpumedfilt_algo_name(algo), t8, tr,
                   algo == auto_algo ? "   <- auto" : "");
        }
        // end-to-end through the public API
        gpu_medfilt2(img.data(), out.data(), W, H, K);   // warm-up (pool, module load)
        const int iters = 5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) gpu_medfilt2(img.data(), out.data(), W, H, K);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        printf("%-4d | %-13s | %12s | %12.3f   (%.0f MPix/s)\n", K, "host->host", "", ms, mpix / ms * 1e3);
        printf("-----+---------------+--------------+-------------\n");
    }

    // transfer cost alone, for reference
    const float h2d = time_ms([&] { CUDA_CHECK(cudaMemcpyAsync(d_in, img.data(), bytes, cudaMemcpyHostToDevice, stream)); }, 5, stream);
    const float d2h = time_ms([&] { CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream)); }, 5, stream);
    printf("\npageable host->device copy: %.3f ms (%.1f GB/s), device->host: %.3f ms (%.1f GB/s)\n",
           h2d, bytes / h2d / 1e6, d2h, bytes / d2h / 1e6);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_in8));
    CUDA_CHECK(cudaFree(d_out));
}

int main(int argc, char** argv)
{
    bool test = true, bench = true;
    int W = 4096, H = 2160, nppK = 0, oneK = 0, oneAlgo = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--test") bench = false;
        else if (a == "--bench") test = false;
        else if (a == "--size" && i + 2 < argc) { W = atoi(argv[++i]); H = atoi(argv[++i]); }
        else if (a == "--npp" && i + 1 < argc) { nppK = atoi(argv[++i]); }
        else if (a == "--one" && i + 2 < argc) { oneK = atoi(argv[++i]); oneAlgo = atoi(argv[++i]); }
        else { fprintf(stderr, "usage: %s [--test|--bench|--npp K|--one K ALGO] [--size W H]\n", argv[0]); return 2; }
    }
    if (oneK) {   // a few launches of one kernel on the benchmark image (for profiling)
        const size_t n = (size_t)W * H, bytes = n * sizeof(float);
        std::vector<float> img = make_image(W, H, 42u, false);
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, img.data(), bytes, cudaMemcpyHostToDevice));
        const float ms = time_ms([&] { CUDA_CHECK(gpumedfilt_device(d_in, d_out, W, H, oneK, 0, oneAlgo)); }, 1, 0, 1);
        printf("K=%d %s: %.3f ms\n", oneK, gpumedfilt_algo_name(oneAlgo), ms);
        return 0;
    }
    if (nppK) {
#ifdef GPUMEDFILT_HAVE_NPP
        run_npp(nppK, W, H);
        return 0;
#else
        fprintf(stderr, "built without NPP support\n");
        return 2;
#endif
    }
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (sm_%d%d, %d SMs)\n\n", p.name, p.major, p.minor, p.multiProcessorCount);

    bool ok = true;
    if (test) ok = run_tests();
    if (bench) run_bench(W, H);
    return ok ? 0 : 1;
}
