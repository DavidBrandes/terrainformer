#include "compute/compute.h"
#include "utils/height_grid.h"
#include "utils/types.h"

#include <driver_types.h>

#include "compute/kernels/kernels.cuh"
#include "compute/types.cuh"
#include "compute/utils.cuh"
#include "kernel.cuh"
#include "setup.cuh"

namespace perf {

constexpr GridConfig::PerlinNoiseConfig PERLIN_NOISE_GRID_CONFIG{
    .seed = 42, .frequency = 6, .octaves = 4, .persistence = 0.5};
constexpr Size SIZE{.width = 8000, .height = 4000};
constexpr GridConfig GRID_CONFIG{.size = SIZE, .initialization = PERLIN_NOISE_GRID_CONFIG};

void launch_smoothstep() {
  HeightGrid height_grid = make_height_grid(GRID_CONFIG);
  GpuBuffer<float> height_grid_buffer = make_gpu_buffer(height_grid);
  compute::ConstrainedGrid heights{compute::Grid{.values = height_grid_buffer.data, .size = SIZE},
                                   HeightGrid::HEIGHT_RANGE};

  Point brush_dab_center = point_from_normalized(SIZE);
  float brush_dab_radius = radius_from_normalized(SIZE);
  BrushDab brush_dab = make_brush_dab(brush_dab_center, brush_dab_radius);

  compute::Region region = compute::aligned_brush_dab_region(brush_dab, SIZE);

  dim3 block_dim(8, 32);
  dim3 grid_dim(compute::ceil_div(region.size.width, 4 * block_dim.x),
                compute::ceil_div(region.size.height, block_dim.y));

  compute::smoothstep<<<grid_dim, block_dim>>>(heights, region.origin, brush_dab);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_marching_squares() {
  HeightGrid height_grid = make_height_grid(GRID_CONFIG);
  GpuBuffer<float> height_grid_buffer = make_gpu_buffer(height_grid);
  compute::CGrid c_heights{.values = height_grid_buffer.data, .size = SIZE};

  GpuBuffer<float4> contour_buffer{(height_grid.size.width - 1) * (height_grid.size.height - 1) * 2};
  GpuBuffer<int> count_buffer{1};
  cudaMemset(count_buffer.data, 0, sizeof(int));

  float threshold = 0;

  GpuBuffer<int2> tmp_coordinates_buffer{contour_buffer.size};

  auto setup = [&]() {
    cudaMemset(count_buffer.data, 0, sizeof(int));
    cudaMemset(contour_buffer.data, 0, contour_buffer.size * sizeof(float4));
    cudaMemset(tmp_coordinates_buffer.data, 0, tmp_coordinates_buffer.size * sizeof(int2));
  };

  auto post = [&](std::string title) {
    int count;

    cudaMemcpy(&count, count_buffer.data, sizeof(int), cudaMemcpyDeviceToHost);

    dim3 block_dim(256);
    dim3 grid_dim(compute::ceil_div(count, block_dim.x));
    perf::marching_squares_part_2<<<grid_dim, block_dim>>>(c_heights, contour_buffer.data, count,
                                                           tmp_coordinates_buffer.data, threshold);

    plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), title);
  };

  auto dflt = [&]() {
    dim3 block_dim(16, 16);
    dim3 grid_dim(compute::ceil_div(SIZE.width - 1, block_dim.x), compute::ceil_div(SIZE.height - 1, block_dim.y));
    perf::marching_squares_part_1<<<grid_dim, block_dim>>>(c_heights, count_buffer.data, contour_buffer.size,
                                                           tmp_coordinates_buffer.data, threshold);
  };

  setup();
  dflt();
}

} // namespace perf

int main() { perf::launch_marching_squares(); }