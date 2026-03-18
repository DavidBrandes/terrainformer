#include "compute/kernels/kernels.h"
#include "compute/parameters.h"

__global__ void smoothstep(Buffers heights, Size grid_size, Vertex offset, Range height_range, BrushDab brush_dab) {
  int col = blockIdx.x * blockDim.x + threadIdx.x + offset.col;
  int row = (blockIdx.y * blockDim.y + threadIdx.y) * SMOOTHSTEP_ROW_COARSE + offset.row;

  for (int phase = 0; phase < SMOOTHSTEP_ROW_COARSE; ++phase) {
    if (col >= grid_size.width || row + phase >= grid_size.height) {
      continue;
    }

    float x_diff = (float)col - brush_dab.circle.center.x;
    float y_diff = (float)(row + phase) - brush_dab.circle.center.y;
    float distance = sqrtf(x_diff * x_diff + y_diff * y_diff);

    if (distance >= brush_dab.circle.radius) {
      continue;
    }

    float factor = 1 - distance / brush_dab.circle.radius;
    float modification = factor * factor * (3 - 2 * factor) * brush_dab.intensity;
    float height = heights.src[(row + phase) * grid_size.width + col] + modification;
    height = max(height_range.min, min(height_range.max, height)); // clamp
    heights.dst[(row + phase) * grid_size.width + col] = height;
  }
}
