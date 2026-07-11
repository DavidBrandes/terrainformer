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

Result compute_contour(std::shared_ptr<MappedGpuResources> resources, std::vector<float> const& thresholds) {
  Segments contours = resources->contours();
  Grid heights = resources->heights().grid;

  CUDA_CHECK(cudaMemset(contours.count, 0, sizeof(int)));

  dim3 block_dim(16, 16);
  dim3 grid_dim(ceil_div(heights.size.width - 1, block_dim.x), ceil_div(heights.size.height - 1, block_dim.y));

  for (float threshold : thresholds) {
    marching_squares<<<grid_dim, block_dim>>>(heights, contours, threshold);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  int contour_segment_count;
  CUDA_CHECK(cudaMemcpy(&contour_segment_count, contours.count, sizeof(int), cudaMemcpyDeviceToHost));

  // TODO (we need to check this)
  if (contour_segment_count > contours.maxCount) {
    std::cerr << std::format("[compute_contour]: Could not write all segments due to limited buffer size. Max possible "
                             "segments: {}, actual segments: {}",
                             contours.maxCount, contour_segment_count)
              << std::endl;
  }

  return Result{.modified = true, .contourSegmentCount = contour_segment_count};
}

Result modify_height(std::shared_ptr<MappedGpuResources> resources, BrushDab brush_dab,
                     std::vector<float> const& thresholds) {
  ConstrainedGrid heights = resources->heights();
  Region region = aligned_brush_dab_region(brush_dab, heights.grid.size);

  if (region.empty()) {
    return Result{.modified = false, .contourSegmentCount = 0};
  }

  dim3 block_dim(8, 32);
  dim3 grid_dim(ceil_div(region.size.width, 4 * block_dim.x), ceil_div(region.size.height, block_dim.y));

  smoothstep<<<grid_dim, block_dim>>>(heights, region.origin, brush_dab);
  CUDA_CHECK(cudaGetLastError());

  // We do syncronization in compute_contour
  return compute_contour(resources, thresholds);
}
} // namespace compute