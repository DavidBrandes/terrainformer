#include "compute/compute.h"
#include "utils/height_grid.h"
#include "utils/types.h"

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
  compute::Grid heights{.values = height_grid_buffer.data, .size = SIZE};

  // GpuBuffer<compute::Segment> contour_buffer{(height_grid.size.width - 1) * (height_grid.size.height - 1) * 2};
  GpuBuffer<int> count_buffer{1};
  cudaMemset(count_buffer.data, 0, sizeof(int));
  // compute::Segments contours{.segments = contour_buffer.data, .count = count_buffer.data};

  float threshold = 0;

  // dim3 block_dim(32, 8);
  // dim3 grid_dim(compute::ceil_div(SIZE.width - 1, block_dim.x), compute::ceil_div(SIZE.height - 1, block_dim.y));

  // perf::marching_squares<<<grid_dim, block_dim>>>(heights, contours, threshold);
  // CUDA_CHECK(cudaGetLastError());
  // CUDA_CHECK(cudaDeviceSynchronize());

  // plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector());

  // GpuBuffer<int> types_buffer_1{(height_grid.size.width - 1) * (height_grid.size.height - 1)};
  GpuBuffer<int> types_buffer_2{(height_grid.size.width - 1) * (height_grid.size.height - 1)};
  // int x1, x2;

  // perf::marching_squares_types<<<grid_dim, block_dim>>>(heights, types_buffer_1.data, count_buffer.data, threshold);
  // CUDA_CHECK(cudaGetLastError());
  // CUDA_CHECK(cudaDeviceSynchronize());

  // cudaMemcpy(&x1, count_buffer.data, sizeof(int), cudaMemcpyDeviceToHost);
  // cudaMemset(count_buffer.data, 0, sizeof(int));

  perf::marching_squares_types_block<<<compute::ceil_div((SIZE.width - 1) * (SIZE.height - 1), 256), 256>>>(
      heights, types_buffer_2.data, count_buffer.data, threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // cudaMemcpy(&x2, count_buffer.data, sizeof(int), cudaMemcpyDeviceToHost);

  // std::cout << "Size " << x1 << ", " << x2 << ", Equal " << (x1 == x2) << std::endl;
}
} // namespace perf

int main() { perf::launch_marching_squares(); }