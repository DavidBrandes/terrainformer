
#include "compute/compute.h"
#include "utils/height_grid.h"
#include "utils/types.h"

#include "compute/kernels/kernels.cuh"
#include "compute/types.cuh"
#include "compute/utils.cuh"
#include "setup.cuh"

namespace perf {

constexpr GridConfig GRID_CONFIG{.size = Size{.width = 8000, .height = 4000}};

void launch_smoothstep() {
  HeightGrid height_grid = make_sinusoidal_height_grid(GRID_CONFIG);
  GpuBuffer<float> height_grid_buffer = make_gpu_buffer(height_grid);
  compute::ConstrainedGrid heights{compute::Grid{.values = height_grid_buffer.data, .size = GRID_CONFIG.size},
                                   HeightGrid::HEIGHT_RANGE};

  Point brush_dab_center = point_from_normalized(GRID_CONFIG);
  float brush_dab_radius = radius_from_normalized(GRID_CONFIG);
  BrushDab brush_dab = make_brush_dab(brush_dab_center, brush_dab_radius);

  compute::Region region = compute::aligned_brush_dab_region(brush_dab, GRID_CONFIG.size);

  dim3 block_dim(16, 16);
  dim3 grid_dim(compute::ceil_div(region.size.width, block_dim.x), compute::ceil_div(region.size.height, block_dim.y));

  compute::smoothstep<<<grid_dim, block_dim>>>(heights, region.origin, brush_dab);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_marching_squares() {
  HeightGrid height_grid = make_sinusoidal_height_grid(GRID_CONFIG);
  GpuBuffer<float> height_grid_buffer = make_gpu_buffer(height_grid);
  compute::Grid heights{.values = height_grid_buffer.data, .size = GRID_CONFIG.size};

  GpuBuffer<compute::Segment> contour_buffer{(height_grid.size.width - 1) * (height_grid.size.height - 1) * 2};
  GpuBuffer<int> count_buffer{1};
  compute::Segments contours{.segments = contour_buffer.data, .count = count_buffer.data};

  float threshold = 0;

  dim3 block_dim(16, 16);
  dim3 grid_dim(compute::ceil_div(heights.size.width - 1, block_dim.x),
                compute::ceil_div(heights.size.height - 1, block_dim.y));

  marching_squares<<<grid_dim, block_dim>>>(heights, contours, threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace perf

int main() { perf::launch_smoothstep(); }