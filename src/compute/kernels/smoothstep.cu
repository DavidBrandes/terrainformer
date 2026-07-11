#include "compute/kernels/kernels.cuh"

namespace compute {

__device__ float apply_brush_dab(float height, float distance, Range height_range, BrushDab brush_dab) {
  float factor = 1 - distance / brush_dab.circle.radius;
  height += factor * factor * (3 - 2 * factor) * brush_dab.intensity;

  return fmaxf(height_range.min, fminf(height_range.max, height)); // clamp
}

__device__ float compute_distance(Vertex position, BrushDab brush_dab) {
  float x_diff = (float)position.col - brush_dab.circle.center.x;
  float y_diff = (float)position.row - brush_dab.circle.center.y;
  return sqrtf(x_diff * x_diff + y_diff * y_diff);
}

__device__ bool is_active(float distance, BrushDab brush_dab) { return distance < brush_dab.circle.radius; }

__global__ void smoothstep(ConstrainedGrid heights, Vertex offset, BrushDab brush_dab) {
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4 + offset.col;
  int row = blockIdx.y * blockDim.y + threadIdx.y + offset.row;

  if (col < heights.grid.size.width && row < heights.grid.size.height) {

    float distance_x = compute_distance(Vertex{.col = col, .row = row}, brush_dab);
    float distance_y = compute_distance(Vertex{.col = col + 1, .row = row}, brush_dab);
    float distance_z = compute_distance(Vertex{.col = col + 2, .row = row}, brush_dab);
    float distance_w = compute_distance(Vertex{.col = col + 3, .row = row}, brush_dab);

    bool active_x = is_active(distance_x, brush_dab);
    bool active_y = is_active(distance_y, brush_dab);
    bool active_z = is_active(distance_z, brush_dab);
    bool active_w = is_active(distance_w, brush_dab);

    if (active_x || active_y || active_z || active_w) {
      float4 values = *(float4*)&(heights.grid[row][col]);

      if (active_x) {
        values.x = apply_brush_dab(values.x, distance_x, heights.range, brush_dab);
      }
      if (active_y) {
        values.y = apply_brush_dab(values.y, distance_y, heights.range, brush_dab);
      }
      if (active_z) {
        values.z = apply_brush_dab(values.z, distance_z, heights.range, brush_dab);
      }
      if (active_w) {
        values.w = apply_brush_dab(values.w, distance_w, heights.range, brush_dab);
      }

      *(float4*)&(heights.grid[row][col]) = values;
    }
  }
}

} // namespace compute