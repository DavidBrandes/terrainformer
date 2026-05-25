#include "compute/compute.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>

#include "compute/kernels/kernels.cuh"
#include "compute/mapped_resources.cuh"
#include "compute/types.cuh"
#include "compute/utils.cuh"
#include "types.cuh"

namespace compute {

Result compute_contour(std::shared_ptr<MappedGpuResources> resources, std::vector<float> const& thresholds) {
  Segments contours = resources->contours();
  Grid heights = resources->heights();

  CUDA_CHECK(cudaMemset(contours.count, 0, sizeof(int)));

  dim3 block_dim(16, 16);
  dim3 grid_dim(ceil_div(heights.size.width - 1, block_dim.x), ceil_div(heights.size.height - 1, block_dim.y));

  std::vector<int> offsets;
  int offset = 0;

  for (float threshold : thresholds) {
    marching_squares<<<grid_dim, block_dim>>>(heights, contours, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&offset, contours.count, sizeof(int), cudaMemcpyDeviceToHost));
    offsets.push_back(offset);
  }

  if (offset > contours.maxCount) {
    std::cerr << std::format("[compute_contour]: Could not write all segments due to limited buffer size. Max possible "
                             "segments: {}, actual segments: {}",
                             contours.maxCount, offset)
              << std::endl;
  }

  return Result{.modified = true, .contourOffsets = offsets};
}

Result modify_height(std::shared_ptr<MappedGpuResources> resources, Range height_range, BrushDab brush_dab,
                     std::vector<float> const& thresholds) {
  Grid heights = resources->heights();

  if (heights.size.width % 4 != 0) {
    throw std::runtime_error("[modify_height]: Grid width needs to be a multiple of 4 for aligned float4 access");
  }

  Region region = aligned_brush_dab_region(brush_dab, heights.size);

  if (region.empty()) {
    return Result{.modified = false, .contourOffsets = std::vector<int>{}};
  }

  dim3 block_dim(8, 32);
  dim3 grid_dim(ceil_div(region.size.width, 4 * block_dim.x), ceil_div(region.size.height, block_dim.y));

  smoothstep<<<grid_dim, block_dim>>>(ConstrainedGrid{heights, height_range}, region.origin, brush_dab);
  CUDA_CHECK(cudaGetLastError());

  return compute_contour(resources, thresholds);
}
} // namespace compute