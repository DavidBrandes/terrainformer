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
  compute::Grid heights{.values = height_grid_buffer.data, .size = SIZE};

  GpuBuffer<compute::Segment> contour_buffer{(height_grid.size.width - 1) * (height_grid.size.height - 1) * 2};
  GpuBuffer<int> count_buffer{1};
  cudaMemset(count_buffer.data, 0, sizeof(int));

  float threshold = 0;

  dim3 block_dim(16, 16);
  dim3 grid_dim(compute::ceil_div(SIZE.width - 1, block_dim.x), compute::ceil_div(SIZE.height - 1, block_dim.y));

  // auto setup_1 = [&]() {
  //   cudaMemset(count_buffer.data, 0, sizeof(int));
  //   cudaMemset(contour_buffer.data, 0, contour_buffer.size * sizeof(compute::Segment));
  // };

  // auto func_1 = [&]() {
  //   perf::marching_squares<<<grid_dim, block_dim>>>(heights, contour_buffer.data, count_buffer.data,
  //                                                   contour_buffer.size, threshold);
  // };

  GpuBuffer<int2> tmp_buffer{contour_buffer.size};

  auto setup_2 = [&]() {
    cudaMemset(count_buffer.data, 0, sizeof(int));
    cudaMemset(contour_buffer.data, 0, contour_buffer.size * sizeof(compute::Segment));
    cudaMemset(tmp_buffer.data, 0, tmp_buffer.size * sizeof(int2));
  };

  auto func_2 = [&]() {
    perf::marching_squares_part_1<<<grid_dim, block_dim>>>(heights, count_buffer.data, contour_buffer.size,
                                                           tmp_buffer.data, threshold);

    int count;
    cudaMemcpy(&count, count_buffer.data, sizeof(int), cudaMemcpyDeviceToHost);

    dim3 block_dim_2(256);
    dim3 grid_dim_2(compute::ceil_div(count, block_dim_2.x));
    perf::marching_squares_part_2<<<grid_dim_2, block_dim_2>>>(heights, contour_buffer.data, count, tmp_buffer.data,
                                                               threshold);
  };

  // auto setup_3 = [&]() {
  //   cudaMemset(count_buffer.data, 0, sizeof(int));
  //   cudaMemset(contour_buffer.data, 0, contour_buffer.size * sizeof(compute::Segment));
  //   cudaMemset(tmp_buffer.data, 0, tmp_buffer.size * sizeof(int2));
  // };

  // auto func_3 = [&]() {
  //   perf::marching_squares_part_1<<<grid_dim, block_dim>>>(heights, count_buffer.data, contour_buffer.size,
  //                                                          tmp_buffer.data, threshold);
  //   dim3 block_dim_2(256);
  //   dim3 grid_dim_2(compute::ceil_div((SIZE.width - 1) * (SIZE.height - 1), block_dim_2.x));
  //   perf::marching_squares_part_2<<<grid_dim_2, block_dim_2>>>(heights, contour_buffer.data, count_buffer.data,
  //                                                              tmp_buffer.data, threshold);
  // };

  // GpuBuffer<int2> tmp_buffer_2{contour_buffer.size};
  // GpuBuffer<Counts> counts_buffer{1};

  // auto setup_4 = [&]() {
  //   cudaMemset(counts_buffer.data, 0, sizeof(Counts));
  //   cudaMemset(contour_buffer.data, 0, contour_buffer.size * sizeof(compute::Segment));
  //   cudaMemset(tmp_buffer.data, 0, tmp_buffer.size * sizeof(int2));
  //   cudaMemset(tmp_buffer_2.data, 0, tmp_buffer.size * sizeof(int2));
  // };

  // auto func_4 = [&]() {
  //   perf::marching_squares_part_1_split<<<grid_dim, block_dim>>>(heights, tmp_buffer.data, tmp_buffer_2.data,
  //                                                                counts_buffer.data, threshold);

  //   Counts counts;
  //   cudaMemcpy(&counts, counts_buffer.data, sizeof(Counts), cudaMemcpyDeviceToHost);

  //   dim3 block_dim_2_single(256);
  //   dim3 grid_dim_2_single(compute::ceil_div(counts.count_1, block_dim_2_single.x));
  //   perf::marching_squares_part_2_single<<<grid_dim_2_single, block_dim_2_single>>>(
  //       heights, contour_buffer.data, tmp_buffer.data, counts.count_1, threshold);

  //   dim3 block_dim_2_double(256);
  //   dim3 grid_dim_2_double(compute::ceil_div(counts.count_2, block_dim_2_double.x));
  //   perf::marching_squares_part_2_double<<<grid_dim_2_double, block_dim_2_double>>>(
  //       heights, contour_buffer.data, tmp_buffer_2.data, counts.count_1, counts.count_2, threshold);
  // };

  // benchmark(func_4, setup_4);
  // plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), "double-split");
  // benchmark(func_3, setup_3);
  // plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), "double-no-copy");
  // benchmark(func_2, setup_2);
  // benchmark(func_1, setup_1);
  // plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), "single");

  // setup_1();
  // func_1();

  setup_2();
  func_2();
  plot(height_grid_buffer.toVector(), SIZE, contour_buffer.toVector(), "double");

  // setup_3();
  // func_3();

  // setup_4();
  // func_4();
}
} // namespace perf

int main() { perf::launch_marching_squares(); }