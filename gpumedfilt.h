#ifndef __GPUMEDFILT_H__
#define __GPUMEDFILT_H__

/* Symbol visibility: the library is built as a shared library (DLL on Windows).
 * - Building the DLL:            GPUMEDFILT_EXPORTS is defined by CMake -> dllexport
 * - Consuming the DLL:           nothing defined                       -> dllimport
 * - Compiling the .cu directly:  define GPUMEDFILT_STATIC              -> plain symbol
 */
#if defined(_WIN32) && !defined(GPUMEDFILT_STATIC)
#  ifdef GPUMEDFILT_EXPORTS
#    define GPUMEDFILT_API __declspec(dllexport)
#  else
#    define GPUMEDFILT_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && !defined(GPUMEDFILT_STATIC)
#  define GPUMEDFILT_API __attribute__((visibility("default")))
#else
#  define GPUMEDFILT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif



/**
 * @brief Apply a 2D median filter to the input image on the GPU.
 *
 * @param input Pointer to the input image data (float array). ROW-MAJOR ORDER. CPU memory.
 * @param output Pointer to the output image data (float array). ROW-MAJOR ORDER. CPU memory.
 * @param width Width of the input image.
 * @param height Height of the input image.
 * @param filter_size Size of the median filter (must be an odd number).
 * 
 * @note The input and output images must be in row-major order. The edge case is always "reflected" (d c b a | a b c d)
 */
GPUMEDFILT_API void gpu_medfilt2(const float* input, float* output, int width, int height, int filter_size);




#ifdef __cplusplus
}
#endif

#endif // __GPUMEDFILT_H__
