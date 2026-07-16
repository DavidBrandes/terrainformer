#include "compute/compute.h"
#include "utils/grid.h"
#include "utils/height_grid.h"
#include "utils/math.h"

#include <driver_types.h>

#include <limits>

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
constexpr int THRESHOLD_COUNT = 100;
constexpr Range THRESHOLD_RANGE{.min = -0.6, .max = 0.6};
constexpr float MAX_COUNT_FRACTION = 0.01;

void launch_smoothstep() {
  HeightGrid height_grid = make_height_grid(GRID_CONFIG);
  GpuBuffer<float> height_grid_buffer = make_gpu_buffer(height_grid);
  compute::Grid grid{.values = height_grid_buffer.data, .size = SIZE};
  compute::ConstrainedGrid constrained_grid{.grid = grid, .range = HeightGrid::HEIGHT_RANGE};

  Point brush_dab_center = point_from_normalized(SIZE);
  float brush_dab_radius = radius_from_normalized(SIZE);
  BrushDab brush_dab = make_brush_dab(brush_dab_center, brush_dab_radius);

  Region region = compute::aligned_brush_dab_region(brush_dab, SIZE);

  dim3 block_dim(8, 32);
  dim3 grid_dim(compute::ceil_div(region.size.width, 4 * block_dim.x),
                compute::ceil_div(region.size.height, block_dim.y));

  compute::smoothstep<<<grid_dim, block_dim>>>(constrained_grid, region.origin, brush_dab);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_marching_squares() {
  HeightGrid height_grid = make_height_grid(GRID_CONFIG);
  GpuBuffer<float> height_grid_buffer = make_gpu_buffer(height_grid);
  compute::CGrid c_heights{.values = height_grid_buffer.data, .size = SIZE};

  int possible_contours_per_threshold = (height_grid.size.width - 1) * (height_grid.size.height - 1) * 2;
  int max_contours_per_threshold = static_cast<int>(possible_contours_per_threshold * MAX_COUNT_FRACTION);

  GpuBuffer<float4> contour_buffer{max_contours_per_threshold * THRESHOLD_COUNT};

  GpuBuffer<int> count_buffer{1};
  cudaMemset(count_buffer.data, 0, count_buffer.bytes());

  GpuBuffer<int2> tmp_coordinates_buffer{max_contours_per_threshold * THRESHOLD_COUNT};
  GpuBuffer<float> tmp_thresholds_buffer{max_contours_per_threshold * THRESHOLD_COUNT};

  std::vector<float> thresholds = linspace(THRESHOLD_RANGE, THRESHOLD_COUNT, Bounds::INCLUDE);
  GpuBuffer<float> thresholds_buffer = GpuBuffer<float>::fromVector(thresholds);

  auto setup = [&]() {
    cudaMemset(count_buffer.data, 0, count_buffer.bytes());
    cudaMemset(contour_buffer.data, 0, contour_buffer.bytes());
    cudaMemset(tmp_coordinates_buffer.data, 0, tmp_coordinates_buffer.bytes());
    cudaMemset(tmp_thresholds_buffer.data, 0, tmp_thresholds_buffer.bytes());
  };

  auto func_base = [&]() {
    dim3 block_dim_1(16, 16, 1);
    dim3 grid_dim_1(compute::ceil_div(SIZE.width - 1, block_dim_1.x), compute::ceil_div(SIZE.height - 1, block_dim_1.y),
                    compute::ceil_div(THRESHOLD_COUNT, block_dim_1.z * COARSE_FACTOR));
    perf::marching_squares_part_1<<<grid_dim_1, block_dim_1>>>(
        c_heights, count_buffer.data, max_contours_per_threshold * THRESHOLD_COUNT, tmp_coordinates_buffer.data,
        thresholds_buffer.data, tmp_thresholds_buffer.data, THRESHOLD_COUNT);

    int count;
    cudaMemcpy(&count, count_buffer.data, sizeof(int), cudaMemcpyDeviceToHost);

    dim3 block_dim_2(256);
    dim3 grid_dim_2(compute::ceil_div(count, block_dim_2.x));
    perf::marching_squares_part_2<<<grid_dim_2, block_dim_2>>>(c_heights, contour_buffer.data, count,
                                                               tmp_coordinates_buffer.data, tmp_thresholds_buffer.data);
  };

  auto func_privatized = [&]() {
    dim3 block_dim_1(16, 16, 1);
    dim3 grid_dim_1(compute::ceil_div(SIZE.width - 1, block_dim_1.x), compute::ceil_div(SIZE.height - 1, block_dim_1.y),
                    compute::ceil_div(THRESHOLD_COUNT, block_dim_1.z * COARSE_FACTOR));
    perf::marching_squares_part_1_privatized<<<grid_dim_1, block_dim_1>>>(
        c_heights, count_buffer.data, max_contours_per_threshold * THRESHOLD_COUNT, tmp_coordinates_buffer.data,
        thresholds_buffer.data, tmp_thresholds_buffer.data, THRESHOLD_COUNT);

    int count;
    cudaMemcpy(&count, count_buffer.data, sizeof(int), cudaMemcpyDeviceToHost);

    dim3 block_dim_2(256);
    dim3 grid_dim_2(compute::ceil_div(count, block_dim_2.x));
    perf::marching_squares_part_2<<<grid_dim_2, block_dim_2>>>(c_heights, contour_buffer.data, count,
                                                               tmp_coordinates_buffer.data, tmp_thresholds_buffer.data);
  };

  setup();
  func_base();
  // plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), "base");

  setup();
  func_privatized();
  // plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), "privatized");
}

} // namespace perf

int main() { perf::launch_marching_squares(); }