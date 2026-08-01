#include <cmath>
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
__device__ int lower_bound(float target, float const* __restrict__ values, int count) {
  int low = 0;
  int high = count;

  while (low < high) {
    int middle = (high + low) / 2;

    if (values[middle] < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }

  return low;
}

__device__ float minimum(float top_left, float top_right, float bottom_right, float bottom_left) {
  return fminf(fminf(top_left, top_right), fminf(bottom_right, bottom_left));
}

__device__ float maximum(float top_left, float top_right, float bottom_right, float bottom_left) {
  return fmaxf(fmaxf(top_left, top_right), fmaxf(bottom_right, bottom_left));
}

__global__ void marching_squares_part_1(compute::CGrid heights, int* __restrict__ count, int max_count,
                                        int2* __restrict__ tmp_coordinates, float const* __restrict__ thresholds,
                                        float* __restrict__ tmp_thresholds, int threshold_count) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  bool active = col < heights.size.width - 1 && row < heights.size.height - 1;

  float top_left, top_right, bottom_right, bottom_left;

  if (active) {
    top_left = heights[row][col];
    top_right = heights[row][col + 1];
    bottom_right = heights[row + 1][col + 1];
    bottom_left = heights[row + 1][col];
  }

  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int stride = blockDim.x * blockDim.y;

  extern __shared__ float thresholds_s[];
  __shared__ int2 tmp_coordinates_s[256];
  __shared__ float tmp_thresholds_s[256];
  __shared__ int count_s;
  __shared__ int offset_s;

  for (int i = thread_id; i < threshold_count; i += stride) {
    thresholds_s[i] = thresholds[i];
  }

  if (thread_id == 0) {
    count_s = 0;
  }

  __syncthreads();

  if (active) {
    float min = minimum(top_left, top_right, bottom_right, bottom_left);
    float max = maximum(top_left, top_right, bottom_right, bottom_left);

    int lower = lower_bound(min, thresholds_s, threshold_count);
    int upper = lower_bound(max, thresholds_s, threshold_count);

    for (int i = lower; i < upper; ++i) {
      float threshold = thresholds_s[i];

      int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);
      int local_count = count_for_type(type);

      if (local_count == 0) {
        continue;
      }

      cuda::atomic_ref<int, cuda::thread_scope_block> segment_count_ref(count_s);
      int block_count = segment_count_ref.fetch_add(local_count, cuda::memory_order_relaxed);

      tmp_coordinates_s[block_count] = int2(col, row);
      tmp_thresholds_s[block_count] = threshold;

      if (local_count == 2) {
        // We let the first entry in the coordinate array compute both output segments.
        // The second one is therefore masked off by setting it to [-1, -1]
        tmp_coordinates_s[block_count + 1] = int2(-1, -1);
      }
    }
  }

  __syncthreads();

  if (thread_id == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(*count);
    offset_s = segment_count_ref.fetch_add(count_s, cuda::memory_order_relaxed);
  }

  __syncthreads();

  int limit = min(max_count - offset_s, count_s);
  for (int i = thread_id; i < limit; i += stride) {
    tmp_coordinates[offset_s + i] = tmp_coordinates_s[i];
    tmp_thresholds[offset_s + i] = tmp_thresholds_s[i];
  }
}

__global__ void marching_squares_part_2(compute::CGrid heights, float4* __restrict__ contours, int count,
                                        int2 const* __restrict__ tmp_coordinates,
                                        float const* __restrict__ tmp_thresholds) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= count) {
    return;
  }

  int2 value = tmp_coordinates[index];
  int col = value.x;
  int row = value.y;

  float threshold = tmp_thresholds[index];

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
    contours[index] = float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                             col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    break;

  case 2:
  case 13:
    contours[index] = float4(col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1),
                             (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
    break;

  case 3:
  case 12:
    contours[index] = float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                             (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
    break;

  case 4:
  case 11:
    contours[index] = float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row,
                             (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
    break;

  case 5:
    if (inside) {
      contours[index] = float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                               col + linear_interpolation_factor(top_left, top_right, threshold), (float)row);
      contours[index + 1] =
          float4(col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1),
                 (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
    } else {
      contours[index] =
          float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                 col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
      contours[index + 1] =
          float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row, (float)(col + 1),
                 row + linear_interpolation_factor(top_right, bottom_right, threshold));
    }
    break;

  case 6:
  case 9:
    contours[index] = float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row,
                             col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    break;

  case 7:
  case 8:
    contours[index] = float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                             col + linear_interpolation_factor(top_left, top_right, threshold), (float)row);
    break;

  case 10:
    if (inside) {
      contours[index] = float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row,
                               (float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold));
      contours[index + 1] =
          float4((float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold),
                 col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    } else {
      contours[index] = float4(col + linear_interpolation_factor(top_left, top_right, threshold), (float)row,
                               (float)col, row + linear_interpolation_factor(top_left, bottom_left, threshold));
      contours[index + 1] =
          float4((float)(col + 1), row + linear_interpolation_factor(top_right, bottom_right, threshold),
                 col + linear_interpolation_factor(bottom_left, bottom_right, threshold), (float)(row + 1));
    }
    break;

  default:
    break;
  }
}
} // namespace perf