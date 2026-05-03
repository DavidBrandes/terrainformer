#include <algorithm>

#include "compute/utils.cuh"
#include "setup.cuh"

namespace perf {
GpuBuffer<float> make_gpu_buffer(HeightGrid const& height_grid) {
  GpuBuffer<float> buffer{height_grid.size.width * height_grid.size.height};
  CUDA_CHECK(cudaMemcpy(buffer.data, height_grid.heights.data(), buffer.size * sizeof(float), cudaMemcpyHostToDevice));

  return buffer;
}

Point point_from_normalized(GridConfig grid_config, float x_normalized, float y_normalized) {
  float x = x_normalized * (grid_config.size.width - 1);
  float y = y_normalized * (grid_config.size.height - 1);

  return Point{x, y};
}

float radius_from_normalized(GridConfig grid_config, float radius_normalized) {
  return (std::min(grid_config.size.height, grid_config.size.width) - 1) * radius_normalized;
}

BrushDab make_brush_dab(Point center, float radius, float intensity) {
  Circle circle{.center = center, .radius = radius};

  return BrushDab{.circle = circle, .intensity = intensity};
}
} // namespace perf