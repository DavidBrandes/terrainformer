FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.2
)
FetchContent_MakeAvailable(cli11)

add_library(perf_common common/setup.cpp common/device.cu)
target_include_directories(perf_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(perf_common PUBLIC terrainformer_lib)
target_compile_options(perf_common PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wsign-compare -Wconversion>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wsign-compare>
)
set_target_properties(perf_common PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(terrainformer_profile profile.cpp)
target_link_libraries(terrainformer_profile PRIVATE perf_common CLI11::CLI11)
target_compile_options(terrainformer_profile PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wsign-compare -Wconversion>
)
set_target_properties(terrainformer_profile PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(terrainformer_benchmark benchmark.cpp)
target_link_libraries(terrainformer_benchmark PRIVATE perf_common CLI11::CLI11)
target_compile_options(terrainformer_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wsign-compare -Wconversion>
)
set_target_properties(terrainformer_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
