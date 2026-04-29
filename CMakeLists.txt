cmake_minimum_required(VERSION 3.18)
project(AmericanOptions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# detect cuda
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    set(HAVE_CUDA TRUE)
else()
    message(WARNING "CUDA compiler not found; CUDA target will be skipped.")
    set(HAVE_CUDA FALSE)
endif()

find_package(OpenMP REQUIRED)

# --- shared core ---
add_library(option_core STATIC
    src/core/black_scholes.cpp
    src/core/quasi_rng.cpp
    src/core/moro_inv_cnd.cpp
    src/core/halton.cpp
    src/core/sobol.cpp
    src/core/scramble.cpp
    src/core/brownian_bridge.cpp
    src/core/halton.cpp
    src/core/sobol.cpp
    src/core/scramble.cpp
    src/core/brownian_bridge.cpp
)
target_include_directories(option_core PUBLIC src)
target_compile_options(option_core PRIVATE -O3 -march=native)

# --- serial cpu ---
add_executable(american_serial
    src/cpu/american_option_serial.cpp
    src/benchmark/main.cpp
)
target_link_libraries(american_serial PRIVATE option_core)
target_compile_options(american_serial PRIVATE -O3 -march=native)
target_compile_definitions(american_serial PRIVATE BACKEND_SERIAL)

# --- openmp ---
add_executable(american_omp
    src/openmp/american_option_omp.cpp
    src/benchmark/main.cpp
)
target_compile_options(american_omp PRIVATE -O3 -march=native)
target_link_libraries(american_omp PRIVATE option_core OpenMP::OpenMP_CXX)
target_compile_definitions(american_omp PRIVATE BACKEND_OMP)

# --- openmp qmc ---
add_executable(american_qmc_omp
    src/openmp/american_option_qmc_omp.cpp
    src/benchmark/main.cpp
)
target_compile_options(american_qmc_omp PRIVATE -fopenmp -O3 -march=native)
target_link_libraries(american_qmc_omp PRIVATE option_core OpenMP::OpenMP_CXX)
target_compile_definitions(american_qmc_omp PRIVATE BACKEND_QMC_OMP)

# --- cuda ---
if(HAVE_CUDA)
    add_executable(american_cuda
        src/cuda/american_option.cu
        src/benchmark/main.cpp
    )
    set_target_properties(american_cuda PROPERTIES CUDA_ARCHITECTURES "native")
    target_compile_options(american_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --ptxas-options=-v>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    )
    target_link_libraries(american_cuda PRIVATE option_core CUDA::cudart)
    target_compile_definitions(american_cuda PRIVATE BACKEND_CUDA)

    add_executable(american_qmc_cuda
        src/cuda/sobol_gpu.cu
        src/cuda/american_option_qmc.cu
        src/benchmark/main.cpp
    )
    set_target_properties(american_qmc_cuda PROPERTIES CUDA_ARCHITECTURES "native")
    target_compile_options(american_qmc_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --ptxas-options=-v -maxrregcount=40>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    )
    target_link_libraries(american_qmc_cuda PRIVATE option_core CUDA::cudart)
    target_compile_definitions(american_qmc_cuda PRIVATE BACKEND_QMC_CUDA)
endif()

# --- validation ---
enable_testing()
add_executable(validate
    tests/validate.cpp
    src/cpu/american_option_serial.cpp
)
target_link_libraries(validate PRIVATE option_core)
target_compile_options(validate PRIVATE -O3 -march=native)
add_test(NAME validation COMMAND validate)