#include "compute/kernels/kernels.cuh"

namespace compute {

__device__ float apply_brush_dab(float height, Vertex position, Range height_range, BrushDab brush_dab) {
  float x_diff = (float)position.col - brush_dab.circle.center.x;
  float y_diff = (float)position.row - brush_dab.circle.center.y;
  float distance = sqrtf(x_diff * x_diff + y_diff * y_diff);

  if (distance >= brush_dab.circle.radius) {
    return height;
  }

  float factor = 1 - distance / brush_dab.circle.radius;
  height += factor * factor * (3 - 2 * factor) * brush_dab.intensity;
  height = max(height_range.min, min(height_range.max, height)); // clamp

  return height;
}

__global__ void smoothstep(ConstrainedGrid heights, Vertex offset, BrushDab brush_dab) {
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4 + offset.col;
  int row = blockIdx.y * blockDim.y + threadIdx.y + offset.row;

  if (col >= heights.size.width || row >= heights.size.height) {
    return;
  }

  float4 values = *(float4*)&(heights[row][col]);

  values.x = apply_brush_dab(values.x, Vertex{.col = col, .row = row}, heights.range, brush_dab);
  values.y = apply_brush_dab(values.y, Vertex{.col = col + 1, .row = row}, heights.range, brush_dab);
  values.z = apply_brush_dab(values.z, Vertex{.col = col + 2, .row = row}, heights.range, brush_dab);
  values.w = apply_brush_dab(values.w, Vertex{.col = col + 3, .row = row}, heights.range, brush_dab);

  *(float4*)&(heights[row][col]) = values;
}

} // namespace compute