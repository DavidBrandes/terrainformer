#include "compute/compute.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>

#include "compute/kernels/kernels.cuh"
#include "compute/mapped_resources.cuh"
#include "compute/types.cuh"
#include "compute/utils.cuh"
#include "types.cuh"

namespace compute {

Result compute_contour(std::shared_ptr<MappedGpuResources> resources) {
  Segments contours = resources->contours();
  CGrid heights = resources->heights();
  Thresholds thresholds = resources->thresholds();

  CUDA_CHECK(cudaMemset(contours.count, 0, sizeof(int)));

  dim3 block_dim(16, 16);
  dim3 grid_dim(ceil_div(heights.size.width - 1, block_dim.x), ceil_div(heights.size.height - 1, block_dim.y));

  marching_squares<<<grid_dim, block_dim, thresholds.count * sizeof(float)>>>(heights, contours, thresholds);
  CUDA_CHECK(cudaGetLastError());

  int contour_count;
  CUDA_CHECK(cudaMemcpy(&contour_count, contours.count, sizeof(int), cudaMemcpyDeviceToHost));

  if (contour_count > contours.maxCount) {
    std::cerr << std::format("[compute_contour]: Could not write all segments due to limited buffer size. Max possible "
                             "segments: {}, actual segments: {}",
                             contours.maxCount, contour_count)
              << std::endl;
  }

  return Result{.modified = true, .contourSegmentCount = std::min(contour_count, contours.maxCount)};
}

Result modify_height(std::shared_ptr<MappedGpuResources> resources, BrushDab brush_dab) {
  ConstrainedGrid heights = resources->constrainedHeights();
  Region region = aligned_brush_dab_region(brush_dab, heights.grid.size);

  if (region.empty()) {
    return Result{.modified = false, .contourSegmentCount = 0};
  }

  dim3 block_dim(8, 32);
  dim3 grid_dim(ceil_div(region.size.width, 4 * block_dim.x), ceil_div(region.size.height, block_dim.y));

  smoothstep<<<grid_dim, block_dim>>>(heights, region.origin, brush_dab);
  CUDA_CHECK(cudaGetLastError());

  // We synchronize in compute_contour
  return compute_contour(resources);
}
} // namespace compute
