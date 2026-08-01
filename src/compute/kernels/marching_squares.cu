#include <cmath>
#include <cuda/atomic>

#include "compute/kernels/kernels.cuh"

namespace compute {

namespace {

struct SubGrid {
  float topLeft;
  float topRight;
  float bottomRight;
  float bottomLeft;
};

constexpr float EPSILON = 1e-6f;

__device__ float linear_interpolation_factor(float start, float end, float value) {
  if (fabsf(end - start) < EPSILON) {
    return 0.5f;
  }

  return (value - start) / (end - start);
}

__device__ int compute_type(SubGrid sub_grid, float threshold) {
  int type = 0;

  if (sub_grid.topLeft > threshold) {
    type |= 8;
  }
  if (sub_grid.topRight > threshold) {
    type |= 4;
  }
  if (sub_grid.bottomRight > threshold) {
    type |= 2;
  }
  if (sub_grid.bottomLeft > threshold) {
    type |= 1;
  }

  return type;
}

// This function computes only a correct saddle point for the cases 5 and 10
__device__ bool compute_inside(int type, SubGrid sub_grid, float threshold) {
  float determinant = (sub_grid.topLeft - threshold) * (sub_grid.bottomRight - threshold) -
                      (sub_grid.topRight - threshold) * (sub_grid.bottomLeft - threshold);

  return type == 10 ? determinant > 0.0f : determinant < 0.0f; // type 5
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

__device__ float minimum(SubGrid sub_grid) {
  return fminf(fminf(sub_grid.topLeft, sub_grid.topRight), fminf(sub_grid.bottomRight, sub_grid.bottomLeft));
}

__device__ float maximum(SubGrid sub_grid) {
  return fmaxf(fmaxf(sub_grid.topLeft, sub_grid.topRight), fmaxf(sub_grid.bottomRight, sub_grid.bottomLeft));
}

__device__ void store_first_contour(int type, SubGrid sub_grid, float threshold, float4* __restrict__ contours) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  bool inside = compute_inside(type, sub_grid, threshold);
  float left_y = row + linear_interpolation_factor(sub_grid.topLeft, sub_grid.bottomLeft, threshold);
  float bottom_x = col + linear_interpolation_factor(sub_grid.bottomLeft, sub_grid.bottomRight, threshold);
  float right_y = row + linear_interpolation_factor(sub_grid.topRight, sub_grid.bottomRight, threshold);
  float top_x = col + linear_interpolation_factor(sub_grid.topLeft, sub_grid.topRight, threshold);

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

__device__ void store_second_contour(int type, SubGrid sub_grid, float threshold, float4* __restrict__ contours) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  bool inside = compute_inside(type, sub_grid, threshold);
  float left_y = row + linear_interpolation_factor(sub_grid.topLeft, sub_grid.bottomLeft, threshold);
  float bottom_x = col + linear_interpolation_factor(sub_grid.bottomLeft, sub_grid.bottomRight, threshold);
  float right_y = row + linear_interpolation_factor(sub_grid.topRight, sub_grid.bottomRight, threshold);
  float top_x = col + linear_interpolation_factor(sub_grid.topLeft, sub_grid.topRight, threshold);

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

} // namespace

__global__ void marching_squares(CGrid heights, Segments contours, Thresholds thresholds) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  bool active = col < heights.size.width - 1 && row < heights.size.height - 1;

  SubGrid sub_grid;

  if (active) {
    sub_grid.topLeft = heights[row][col];
    sub_grid.topRight = heights[row][col + 1];
    sub_grid.bottomRight = heights[row + 1][col + 1];
    sub_grid.bottomLeft = heights[row + 1][col];
  }

  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int stride = blockDim.x * blockDim.y;

  extern __shared__ float thresholds_s[];

  for (int i = thread_id; i < thresholds.count; i += stride) {
    thresholds_s[i] = thresholds.values[i];
  }

  __syncthreads();

  if (!active) {
    return;
  }

  float min = minimum(sub_grid);
  float max = maximum(sub_grid);

  int lower = lower_bound(min, thresholds_s, thresholds.count);
  int upper = lower_bound(max, thresholds_s, thresholds.count);

  for (int i = lower; i < upper; ++i) {
    float threshold = thresholds_s[i];

    int type = compute_type(sub_grid, threshold);

    if (type == 0 || type == 15) {
      continue;
    }

    cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(*contours.count);
    int offset = segment_count_ref.fetch_add(1, cuda::memory_order_relaxed);

    if (offset >= contours.maxCount) {
      continue;
    }

    store_first_contour(type, sub_grid, threshold, contours.values + offset);

    if (type != 5 && type != 10) {
      continue;
    }

    offset = segment_count_ref.fetch_add(1, cuda::memory_order_relaxed);

    if (offset >= contours.maxCount) {
      continue;
    }

    store_second_contour(type, sub_grid, threshold, contours.values + offset);
  }
}

} // namespace compute