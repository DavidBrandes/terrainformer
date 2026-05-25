find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(terrainformer_perf main.cu setup.cu kernel.cu)
target_include_directories(terrainformer_perf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(terrainformer_perf PRIVATE
    terrainformer_lib
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)
target_compile_options(terrainformer_perf PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wsign-compare -Wconversion>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wsign-compare>
)
set_target_properties(terrainformer_perf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)