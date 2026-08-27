cmake_minimum_required(VERSION 3.24)

# Build for the GPU in this machine by default; override with e.g. -DCMAKE_CUDA_ARCHITECTURES="86;89;90".
# This must run before project(): CMake fixes the architectures while it detects the CUDA compiler.
# CMake's own "native" is not expanded by the Visual Studio generator (the project silently falls back
# to nvcc's default sm_75), so the local GPU is queried here with nvcc's helper instead.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "" OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "native")
    find_program(_nvcc_device_query __nvcc_device_query HINTS ENV CUDA_PATH ENV CUDA_HOME PATH_SUFFIXES bin)
    set(_native_arch "")
    if(_nvcc_device_query)
        execute_process(COMMAND "${_nvcc_device_query}" OUTPUT_VARIABLE _native_arch RESULT_VARIABLE _query_result
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REGEX REPLACE "[ \n\r]+" ";" _native_arch "${_native_arch}")
        if(NOT _query_result EQUAL 0 OR NOT "${_native_arch}" MATCHES "^[0-9]+(;[0-9]+)*$")
            set(_native_arch "")
        endif()
    endif()
    if(_native_arch)
        set(CMAKE_CUDA_ARCHITECTURES "${_native_arch}")
    else()
        message(WARNING "Could not detect the local GPU architecture; using CMAKE_CUDA_ARCHITECTURES=native "
                        "(ignored by the Visual Studio generator). Pass -DCMAKE_CUDA_ARCHITECTURES=<sm> explicitly.")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

project(gpumedfilt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)   # DLL carries its own CUDA runtime -> no cudart DLL dependency

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# ---- the shared library -------------------------------------------------------------------------
add_library(gpumedfilt SHARED gpumedfilt.cu gpumedfilt.h gpumedfilt_internal.h)
target_compile_definitions(gpumedfilt PRIVATE GPUMEDFILT_EXPORTS)
target_include_directories(gpumedfilt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gpumedfilt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(gpumedfilt PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# ---- tests / benchmark --------------------------------------------------------------------------
option(GPUMEDFILT_BUILD_TESTS "Build the correctness test + benchmark executable" ON)
option(GPUMEDFILT_BENCH_NPP   "Allow benchmarking against NPP's nppiFilterMedianBorder_32f_C1R (medfilt_test --npp K)" ON)

if(GPUMEDFILT_BUILD_TESTS)
    # The harness compiles gpumedfilt.cu directly (GPUMEDFILT_STATIC) so it can drive the
    # individual kernels through the private API.
    add_executable(medfilt_test test/medfilt_test.cu gpumedfilt.cu)
    target_compile_definitions(medfilt_test PRIVATE GPUMEDFILT_STATIC)
    target_include_directories(medfilt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(medfilt_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    if(GPUMEDFILT_BENCH_NPP)
        target_compile_definitions(medfilt_test PRIVATE GPUMEDFILT_HAVE_NPP)
        target_link_libraries(medfilt_test PRIVATE CUDA::nppif CUDA::nppc)
    endif()

    enable_testing()
    add_test(NAME correctness COMMAND medfilt_test --test)
endif()
