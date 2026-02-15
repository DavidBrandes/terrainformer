#include "compute/kernels/kernels.h"
#include "compute/parameters.h"

__global__ void smoothstep(float* heights, Size grid_size, Range height_range, BrushDab brush_dab) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < grid_size.width * grid_size.height; i += stride) {
    float row = static_cast<float>(i / grid_size.width);
    float column = static_cast<float>(i % grid_size.width);

    float distance = sqrtf(powf(brush_dab.circle.center.x - column, 2) + powf(brush_dab.circle.center.y - row, 2));

    if (distance < brush_dab.circle.radius) {
      float value = 1.0f - distance / brush_dab.circle.radius;
      float result = value * value * (3.0f - 2.0f * value) * brush_dab.intensity;
      float new_height = heights[i] + result;

      heights[i] = max(height_range.min, min(height_range.max, new_height)); // clamp
    }
  }
}