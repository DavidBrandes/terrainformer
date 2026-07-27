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

// __device__ bool compute_inside(float top_left, float top_right, float bottom_right, float bottom_left, float
// threshold,
//                                int type) {
//   float const determinant =
//       (top_left - threshold) * (bottom_right - threshold) - (top_right - threshold) * (bottom_left - threshold);

//   return type == 10 ? determinant > 0.0f : determinant < 0.0f; // type 5
// }

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

__device__ void store_first_contour(int type, float top_left, float top_right, float bottom_right, float bottom_left,
                                    float threshold, float4* __restrict__ contours) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);
  float left_y = row + linear_interpolation_factor(top_left, bottom_left, threshold);
  float bottom_x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold);
  float right_y = row + linear_interpolation_factor(top_right, bottom_right, threshold);
  float top_x = col + linear_interpolation_factor(top_left, top_right, threshold);

  switch (type) {
  case 1:
  case 14:
    *contours = float4((float)col, left_y, bottom_x, (float)(row + 1));
    break;

  case 2:
  case 13:
    *contours = float4(bottom_x, (float)(row + 1), (float)(col + 1), right_y);
    break;

  case 3:
  case 12:
    *contours = float4((float)col, left_y, (float)(col + 1), right_y);
    break;

  case 4:
  case 11:
    *contours = float4(top_x, (float)row, (float)(col + 1), right_y);
    break;

  case 5:
    if (inside) {
      *contours = float4((float)col, left_y, top_x, (float)row);
    } else {
      *contours = float4((float)col, left_y, bottom_x, (float)(row + 1));
    }
    break;

  case 6:
  case 9:
    *contours = float4(top_x, (float)row, bottom_x, (float)(row + 1));
    break;

  case 7:
  case 8:
    *contours = float4((float)col, left_y, top_x, (float)row);
    break;

  case 10:
    if (inside) {
      *contours = float4(top_x, (float)row, (float)(col + 1), right_y);
    } else {
      *contours = float4(top_x, (float)row, (float)col, left_y);
    }
    break;

  default:
    break;
  }
}

__device__ void store_second_contour(int type, float top_left, float top_right, float bottom_right, float bottom_left,
                                     float threshold, float4* __restrict__ contours) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);
  float left_y = row + linear_interpolation_factor(top_left, bottom_left, threshold);
  float bottom_x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold);
  float right_y = row + linear_interpolation_factor(top_right, bottom_right, threshold);
  float top_x = col + linear_interpolation_factor(top_left, top_right, threshold);

  if (type == 5) {
    if (inside) {
      *contours = float4(bottom_x, (float)(row + 1), (float)(col + 1), right_y);
    } else {
      *contours = float4(top_x, (float)row, (float)(col + 1), right_y);
    }
  }

  if (type == 10) {
    if (inside) {
      *contours = float4((float)col, left_y, bottom_x, (float)(row + 1));
    } else {
      *contours = float4((float)(col + 1), right_y, bottom_x, (float)(row + 1));
    }
  }
}

__global__ void marching_squares(compute::CGrid heights, int* __restrict__ count, int max_count,
                                 float const* __restrict__ thresholds, float4* __restrict__ contours,
                                 int threshold_count) {
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

  for (int i = thread_id; i < threshold_count; i += stride) {
    thresholds_s[i] = thresholds[i];
  }

  __syncthreads();

  if (!active) {
    return;
  }

  float min = minimum(top_left, top_right, bottom_right, bottom_left);
  float max = maximum(top_left, top_right, bottom_right, bottom_left);

  int lower = lower_bound(min, thresholds_s, threshold_count);
  int upper = lower_bound(max, thresholds_s, threshold_count);

  for (int i = lower; i < upper; ++i) {
    float threshold = thresholds_s[i];

    int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);

    if (type == 0 || type == 15) {
      continue;
    }

    cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(*count);
    int global_count = segment_count_ref.fetch_add(1, cuda::memory_order_relaxed);

    if (global_count >= max_count) {
      continue;
    }

    store_first_contour(type, top_left, top_right, bottom_right, bottom_left, threshold, contours + global_count);

    if (type != 5 && type != 10) {
      continue;
    }

    global_count = segment_count_ref.fetch_add(1, cuda::memory_order_relaxed);

    if (global_count >= max_count) {
      continue;
    }

    store_second_contour(type, top_left, top_right, bottom_right, bottom_left, threshold, contours + global_count);
  }
}
} // namespace perf