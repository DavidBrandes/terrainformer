#include <cmath>
#include <cub/cub.cuh>
#include <cuda/atomic>

#include "kernel.cuh"

namespace perf {

constexpr float EPSILON = 1e-6f;

__device__ float linear_interpolation_factor(float start, float end, float value) {
  if (fabsf(end - start) < EPSILON) {
    return 0.5f;
  }

  return (value - start) / (end - start);
}

__device__ int compute_type(float top_left, float top_right, float bottom_right, float bottom_left, float threshold) {
  int type = 0;

  if (top_left > threshold) {
    type |= 8;
  }
  if (top_right > threshold) {
    type |= 4;
  }
  if (bottom_right > threshold) {
    type |= 2;
  }
  if (bottom_left > threshold) {
    type |= 1;
  }

  return type;
}

// This function computes only a correct saddle point for the cases 5 and 10
__device__ bool compute_inside(float top_left, float top_right, float bottom_right, float bottom_left,
                               float threshold) {
  float denominator = top_left - top_right + bottom_right - bottom_left;
  float u = (top_left - bottom_left) / denominator;
  float v = (top_left - top_right) / denominator;

  float saddle =
      top_left * (1 - u) * (1 - v) + top_right * u * (1 - v) + bottom_right * u * v + bottom_left * (1 - u) * v;

  return saddle > threshold;
}

__device__ int count_for_type(int type) {
  switch (type) {
  case 0:
  case 15:
    return 0;
  case 5:
  case 10:
    return 2;
  default:
    return 1;
  }
}

__global__ void marching_squares_part_1(compute::CGrid heights, int* __restrict__ count, int max_count,
                                        int2* __restrict__ tmp, float threshold) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= heights.size.width - 1 || row >= heights.size.height - 1) {
    return;
  }

  float top_left = heights[row][col];
  float top_right = heights[row][col + 1];
  float bottom_right = heights[row + 1][col + 1];
  float bottom_left = heights[row + 1][col];

  int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);
  int local_count = count_for_type(type);

  if (local_count == 0) {
    return;
  }

  cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(*count);
  int global_count = segment_count_ref.fetch_add(local_count, cuda::memory_order_relaxed);

  // TODO this should be refined
  if (local_count + global_count > max_count) {
    return;
  }

  tmp[global_count] = int2(col, row);

  if (local_count == 2) {
    // We let the first entry in the coordinate array compute both output segments.
    // The second one is therefore masked off by setting it to [-1, -1]
    tmp[global_count + 1] = int2(-1, -1);
  }
}

__global__ void marching_squares_part_2(compute::CGrid heights, float4* __restrict__ contours, int count,
                                        int2 const* __restrict__ tmp, float threshold, int offset) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= count) {
    return;
  }

  int2 value = tmp[index];

  int col = value.x;
  int row = value.y;

  // In the case of two output segments, we reserve two indices in the coordinate and output arrays.
  // Since the first of them computes both entries, we mark the second one with [-1, -1] to show that
  // computation will happen elsewhere.
  if (col == -1) {
    return;
  }

  float top_left = heights[row][col];
  float top_right = heights[row][col + 1];
  float bottom_right = heights[row + 1][col + 1];
  float bottom_left = heights[row + 1][col];

  int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);
  bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);

  switch (type) {
  case 0:
  case 15:
    break;

  case 1:
  case 14:
    contours[index + offset] =
        float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
               col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    break;

  case 2:
  case 13:
    contours[index + offset] =
        float4(col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1),
               (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
    break;

  case 3:
  case 12:
    contours[index + offset] =
        float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold), (float)(col + 1),
               row + linear_interpolation_factor(top_right, bottom_right, threshold));
    break;

  case 4:
  case 11:
    contours[index + offset] =
        float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row, (float)(col + 1),
               row + linear_interpolation_factor(top_right, bottom_right, threshold));
    break;

  case 5:
    if (inside) {
      contours[index + offset] = float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                                        col + linear_interpolation_factor(top_left, top_right, threshold), (float)row);
      contours[index + offset + 1] =
          float4(col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1),
                 (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
    } else {
      contours[index + offset] =
          float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                 col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
      contours[index + offset + 1] =
          float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row, (float)(col + 1),
                 row + linear_interpolation_factor(top_right, bottom_right, threshold));
    }
    break;

  case 6:
  case 9:
    contours[index + offset] =
        float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row,
               col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    break;

  case 7:
  case 8:
    contours[index + offset] = float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                                      col + linear_interpolation_factor(top_left, top_right, threshold), (float)row);
    break;

  case 10:
    if (inside) {
      contours[index + offset] =
          float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row, (float)(col + 1),
                 row + linear_interpolation_factor(top_right, bottom_right, threshold));
      contours[index + offset + 1] =
          float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                 col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    } else {
      contours[index + offset] =
          float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row, (float)col,
                 row + linear_interpolation_factor(top_left, bottom_left, threshold));
      contours[index + offset + 1] =
          float4((float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold),
                 col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    }
    break;

  default:
    break;
  }
}

} // namespace perf