cmake_minimum_required(VERSION 3.24)
project(terrainformer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.8
    SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl STATIC REPRODUCIBLE LOADER API gl:core=3.3)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.5
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

set(LIB_SOURCES
    src/app/app.cpp
    src/controller/scene.cpp
    src/ui/window.cpp
    src/ui/gui.cpp
    src/graphics/renderer.cpp
    src/graphics/scene.cpp
    src/graphics/layers/circle.cpp
    src/graphics/layers/contour.cpp
    src/graphics/layers/map.cpp
    src/graphics/layers/vertex.cpp
    src/utils/height_grid.cpp
    src/utils/config.cpp
    src/compute/compute.cu
    src/compute/resources.cu
    src/compute/kernels/marching_squares.cu
    src/compute/kernels/smoothstep.cu
)

add_library(terrainformer_lib STATIC ${LIB_SOURCES})

target_link_libraries(terrainformer_lib
    PUBLIC
    OpenGL::GL
    CUDA::cudart
    glfw
    glad_gl
    imgui
    tomlplusplus::tomlplusplus
)

target_include_directories(terrainformer_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(terrainformer_lib
    PUBLIC
    CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config.toml"
    SHADERS_DIR="${CMAKE_SOURCE_DIR}/src/graphics/shaders"
)

target_compile_options(terrainformer_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wsign-compare -Wconversion>
)

target_compile_options(terrainformer_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall,-Wextra,-Wsign-compare
        #-Wdefault-stream-launch  TODO check if needed
    >
)

set_target_properties(terrainformer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(terrainformer src/main.cpp)
target_link_libraries(terrainformer PRIVATE terrainformer_lib)

option(BUILD_PERF "Build performance tools (benchmarks, profiling)" OFF)
if(BUILD_PERF)
    add_subdirectory(perf)
endif()