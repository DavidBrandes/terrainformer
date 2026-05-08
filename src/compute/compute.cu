#include "compute/compute.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "compute/kernels/kernels.cuh"
#include "compute/mapped_resources.cuh"
#include "compute/types.cuh"
#include "compute/utils.cuh"
#include "types.cuh"

namespace compute {

Result compute_contour(std::shared_ptr<MappedGpuResources> resources, float threshold) {
  Segments contours = resources->contours();
  Grid heights = resources->heights();

  CUDA_CHECK(cudaMemset(contours.count, 0, sizeof(int)));

  dim3 block_dim(16, 16);
  dim3 grid_dim(ceil_div(heights.size.width - 1, block_dim.x), ceil_div(heights.size.height - 1, block_dim.y));

  marching_squares<<<grid_dim, block_dim>>>(heights, contours, threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int segment_count;
  CUDA_CHECK(cudaMemcpy(&segment_count, contours.count, sizeof(int), cudaMemcpyDeviceToHost));

  return Result{.contourSegmentCount = segment_count};
}

Result modify_height(std::shared_ptr<MappedGpuResources> resources, Range height_range, BrushDab brush_dab,
                     float contour_threshold) {
  Grid heights = resources->heights();

  if (heights.size.width % 4 != 0) {
    throw std::runtime_error("Grid width needs to be a multiple of 4 for aligned float4 access");
  }

  Region region = aligned_brush_dab_region(brush_dab, heights.size);

  if (region.empty()) {
    return Result{.contourSegmentCount = 0};
  }

  dim3 block_dim(8, 32);
  dim3 grid_dim(ceil_div(region.size.width, 4 * block_dim.x), ceil_div(region.size.height, block_dim.y));

  smoothstep<<<grid_dim, block_dim>>>(ConstrainedGrid{heights, height_range}, region.origin, brush_dab);
  CUDA_CHECK(cudaGetLastError());

  Segments contours = resources->contours();

  CUDA_CHECK(cudaMemset(contours.count, 0, sizeof(int)));

  block_dim = dim3(16, 16);
  grid_dim = dim3(ceil_div(heights.size.width - 1, block_dim.x), ceil_div(heights.size.height - 1, block_dim.y));

  marching_squares<<<grid_dim, block_dim>>>(heights, contours, contour_threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int segment_count;
  CUDA_CHECK(cudaMemcpy(&segment_count, contours.count, sizeof(int), cudaMemcpyDeviceToHost));

  return Result{.contourSegmentCount = segment_count};
}
} // namespace compute