FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(terrainformer_benchmark benchmark.cpp)
target_link_libraries(terrainformer_benchmark PRIVATE terrainformer_lib benchmark::benchmark benchmark::benchmark_main)
