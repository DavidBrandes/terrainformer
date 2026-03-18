#include "compute/compute.h"
#include "compute/kernels/kernels.h"
#include "compute/parameters.h"
#include "kernels/kernels.h"
#include "parameters.h"

#include <stdio.h>

#include <algorithm>
#include <cmath>

Region brush_dab_region(BrushDab brush_dab, Size grid_size) {
  // Only grid vertices strictly inside the brush dab circle are selected
  int col_start = std::max(0, static_cast<int>(std::floor(brush_dab.circle.center.x - brush_dab.circle.radius + 1)));
  int col_end = std::min(static_cast<int>(std::ceil(brush_dab.circle.center.x + brush_dab.circle.radius - 1)),
                         grid_size.width - 1);
  int row_start = std::max(0, static_cast<int>(std::floor(brush_dab.circle.center.y - brush_dab.circle.radius + 1)));
  int row_end = std::min(static_cast<int>(std::ceil(brush_dab.circle.center.y + brush_dab.circle.radius - 1)),
                         grid_size.height - 1);

  Vertex origin{.col = col_start, .row = row_start};
  Size size{.width = col_end - col_start + 1, .height = row_end - row_start + 1};

  return Region{
      .origin = origin,
      .size = size,
  };
}

void compute_contour(std::unique_ptr<MappedGpuResources> resources, Size grid_size, float threshold) {
  marching_squares<<<1, 1>>>(resources->heights(), resources->contours(), grid_size, threshold);
  cudaDeviceSynchronize();
}

void modify_height(std::unique_ptr<MappedGpuResources> resources, Size grid_size, Range height_range,
                   BrushDab brush_dab, float contour_threshold) {

  Region region = brush_dab_region(brush_dab, grid_size);

  if (region.empty()) {
    return;
  }

  Buffers heights{.src = resources->heights(), .dst = resources->heights()};

  dim3 block_dim(32, 4);
  dim3 grid_dim((region.size.width + block_dim.x - 1) / block_dim.x,
                (region.size.height + block_dim.y * SMOOTHSTEP_ROW_COARSE - 1) / (block_dim.y * SMOOTHSTEP_ROW_COARSE));
  smoothstep<<<grid_dim, block_dim>>>(heights, grid_size, region.origin, height_range, brush_dab);
  cudaDeviceSynchronize();

  // marching_squares<<<1, 1>>>(resources->heights(), resources->contours(), grid_size, contour_threshold);
  // cudaDeviceSynchronize();
}
