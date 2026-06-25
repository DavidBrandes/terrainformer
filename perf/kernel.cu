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
  case 1:
  case 2:
  case 3:
  case 4:
  case 6:
  case 7:
  case 8:
  case 9:
  case 11:
  case 12:
  case 13:
  case 14:
    return 1;
  case 5:
  case 10:
    return 2;
  default:
    return 0;
  }
}

// __global__ void marching_squares(compute::Grid heights, compute::Segment* contours, int* count, int max_count,
//                                  float threshold) {
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   int row = blockIdx.y * blockDim.y + threadIdx.y;

//   if (col >= heights.size.width - 1 || row >= heights.size.height - 1) {
//     return;
//   }

//   float top_left = heights[row][col];
//   float top_right = heights[row][col + 1];
//   float bottom_right = heights[row + 1][col + 1];
//   float bottom_left = heights[row + 1][col];

//   bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);
//   int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);

//   Point top{.x = col + linear_interpolation_factor(top_left, top_right, threshold), .y = (float)row};
//   Point right{.x = (float)(col + 1), .y = row + linear_interpolation_factor(top_right, bottom_right, threshold)};
//   Point bottom{.x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold), .y = (float)(row + 1)};
//   Point left{.x = (float)col, .y = row + linear_interpolation_factor(top_left, bottom_left, threshold)};

//   int local_count = count_for_type(type);
//   cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(*count);
//   int global_count = segment_count_ref.fetch_add(local_count, cuda::memory_order_relaxed);

//   if (local_count + global_count > max_count) {
//     return;
//   }

//   switch (type) {
//   case 0:
//   case 15:
//     break;

//   case 1:
//   case 14:
//     contours[global_count] = compute::Segment{.start = left, .end = bottom};
//     break;

//   case 2:
//   case 13:
//     contours[global_count] = compute::Segment{.start = bottom, .end = right};
//     break;

//   case 3:
//   case 12:
//     contours[global_count] = compute::Segment{.start = left, .end = right};
//     break;

//   case 4:
//   case 11:
//     contours[global_count] = compute::Segment{.start = top, .end = right};
//     break;

//   case 5:
//     if (inside) {
//       contours[global_count] = compute::Segment{.start = left, .end = top};
//       contours[global_count + 1] = compute::Segment{.start = bottom, .end = right};
//     } else {
//       contours[global_count] = compute::Segment{.start = left, .end = bottom};
//       contours[global_count + 1] = compute::Segment{.start = top, .end = right};
//     }
//     break;

//   case 6:
//   case 9:
//     contours[global_count] = compute::Segment{.start = top, .end = bottom};
//     break;

//   case 7:
//   case 8:
//     contours[global_count] = compute::Segment{.start = left, .end = top};
//     break;

//   case 10:
//     if (inside) {
//       contours[global_count] = compute::Segment{.start = top, .end = right};
//       contours[global_count + 1] = compute::Segment{.start = left, .end = bottom};
//     } else {
//       contours[global_count] = compute::Segment{.start = top, .end = left};
//       contours[global_count + 1] = compute::Segment{.start = right, .end = bottom};
//     }
//     break;

//   default:
//     break;
//   }
// }

__global__ void marching_squares_part_1(compute::Grid heights, int* count, int max_count, int2* tmp, float threshold) {
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

  if (local_count + global_count > max_count) {
    return;
  }

  if (local_count == 1) {
    tmp[global_count] = int2(col, row);
  }

  if (local_count == 2) {
    tmp[global_count + 1] = int2(-1, -1);
  }
}

__global__ void marching_squares_part_2(compute::Grid heights, compute::Segment* contours, int count, int2* tmp,
                                        float threshold) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > count) {
    return;
  }

  int2 value = tmp[index];

  int col = value.x;
  int row = value.y;

  if (col == -1) {
    return;
  }

  float top_left = heights[row][col];
  float top_right = heights[row][col + 1];
  float bottom_right = heights[row + 1][col + 1];
  float bottom_left = heights[row + 1][col];

  bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);
  int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);

  Point top{.x = col + linear_interpolation_factor(top_left, top_right, threshold), .y = (float)row};
  Point right{.x = (float)(col + 1), .y = row + linear_interpolation_factor(top_right, bottom_right, threshold)};
  Point bottom{.x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold), .y = (float)(row + 1)};
  Point left{.x = (float)col, .y = row + linear_interpolation_factor(top_left, bottom_left, threshold)};

  switch (type) {
  case 0:
  case 15:
    break;

  case 1:
  case 14:
    contours[index] = compute::Segment{.start = left, .end = bottom};
    break;

  case 2:
  case 13:
    contours[index] = compute::Segment{.start = bottom, .end = right};
    break;

  case 3:
  case 12:
    contours[index] = compute::Segment{.start = left, .end = right};
    break;

  case 4:
  case 11:
    contours[index] = compute::Segment{.start = top, .end = right};
    break;

  case 5:
    if (inside) {
      contours[index] = compute::Segment{.start = left, .end = top};
      contours[index + 1] = compute::Segment{.start = bottom, .end = right};
    } else {
      contours[index] = compute::Segment{.start = left, .end = bottom};
      contours[index + 1] = compute::Segment{.start = top, .end = right};
    }
    break;

  case 6:
  case 9:
    contours[index] = compute::Segment{.start = top, .end = bottom};
    break;

  case 7:
  case 8:
    contours[index] = compute::Segment{.start = left, .end = top};
    break;

  case 10:
    if (inside) {
      contours[index] = compute::Segment{.start = top, .end = right};
      contours[index + 1] = compute::Segment{.start = left, .end = bottom};
    } else {
      contours[index] = compute::Segment{.start = top, .end = left};
      contours[index + 1] = compute::Segment{.start = right, .end = bottom};
    }
    break;

  default:
    break;
  }
}

// __global__ void marching_squares_part_2(compute::Grid heights, compute::Segment* contours, int* count, int2* tmp,
//                                         float threshold) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index > *count) {
//     return;
//   }

//   int2 value = tmp[index];

//   int col = value.x;
//   int row = value.y;

//   if (col == -1) {
//     return;
//   }

//   float top_left = heights[row][col];
//   float top_right = heights[row][col + 1];
//   float bottom_right = heights[row + 1][col + 1];
//   float bottom_left = heights[row + 1][col];

//   bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);
//   int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);

//   Point top{.x = col + linear_interpolation_factor(top_left, top_right, threshold), .y = (float)row};
//   Point right{.x = (float)(col + 1), .y = row + linear_interpolation_factor(top_right, bottom_right, threshold)};
//   Point bottom{.x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold), .y = (float)(row + 1)};
//   Point left{.x = (float)col, .y = row + linear_interpolation_factor(top_left, bottom_left, threshold)};

//   switch (type) {
//   case 0:
//   case 15:
//     break;

//   case 1:
//   case 14:
//     contours[index] = compute::Segment{.start = left, .end = bottom};
//     break;

//   case 2:
//   case 13:
//     contours[index] = compute::Segment{.start = bottom, .end = right};
//     break;

//   case 3:
//   case 12:
//     contours[index] = compute::Segment{.start = left, .end = right};
//     break;

//   case 4:
//   case 11:
//     contours[index] = compute::Segment{.start = top, .end = right};
//     break;

//   case 5:
//     if (inside) {
//       contours[index] = compute::Segment{.start = left, .end = top};
//       contours[index + 1] = compute::Segment{.start = bottom, .end = right};
//     } else {
//       contours[index] = compute::Segment{.start = left, .end = bottom};
//       contours[index + 1] = compute::Segment{.start = top, .end = right};
//     }
//     break;

//   case 6:
//   case 9:
//     contours[index] = compute::Segment{.start = top, .end = bottom};
//     break;

//   case 7:
//   case 8:
//     contours[index] = compute::Segment{.start = left, .end = top};
//     break;

//   case 10:
//     if (inside) {
//       contours[index] = compute::Segment{.start = top, .end = right};
//       contours[index + 1] = compute::Segment{.start = left, .end = bottom};
//     } else {
//       contours[index] = compute::Segment{.start = top, .end = left};
//       contours[index + 1] = compute::Segment{.start = right, .end = bottom};
//     }
//     break;

//   default:
//     break;
//   }
// }

// __global__ void marching_squares_part_1_split(compute::Grid heights, int2* tmp_1, int2* tmp_2, Counts* counts,
//                                               float threshold) {
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   int row = blockIdx.y * blockDim.y + threadIdx.y;

//   if (col >= heights.size.width - 1 || row >= heights.size.height - 1) {
//     return;
//   }

//   float top_left = heights[row][col];
//   float top_right = heights[row][col + 1];
//   float bottom_right = heights[row + 1][col + 1];
//   float bottom_left = heights[row + 1][col];

//   int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);
//   int local_count = count_for_type(type);

//   if (local_count == 1) {
//     cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(counts->count_1);
//     int global_count = segment_count_ref.fetch_add(local_count, cuda::memory_order_relaxed);
//     tmp_1[global_count] = int2(col, row);
//   }

//   if (local_count == 2) {
//     cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(counts->count_2);
//     int global_count = segment_count_ref.fetch_add(local_count, cuda::memory_order_relaxed);
//     tmp_2[global_count] = int2(col, row);
//   }
// }
// __global__ void marching_squares_part_2_single(compute::Grid heights, compute::Segment* contours, int2* tmp_1,
//                                                int count_1, float threshold) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index > count_1) {
//     return;
//   }

//   int2 value = tmp_1[index];

//   int col = value.x;
//   int row = value.y;

//   float top_left = heights[row][col];
//   float top_right = heights[row][col + 1];
//   float bottom_right = heights[row + 1][col + 1];
//   float bottom_left = heights[row + 1][col];

//   int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);

//   Point top{.x = col + linear_interpolation_factor(top_left, top_right, threshold), .y = (float)row};
//   Point right{.x = (float)(col + 1), .y = row + linear_interpolation_factor(top_right, bottom_right, threshold)};
//   Point bottom{.x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold), .y = (float)(row + 1)};
//   Point left{.x = (float)col, .y = row + linear_interpolation_factor(top_left, bottom_left, threshold)};

//   switch (type) {
//   case 0:
//   case 15:
//     break;

//   case 1:
//   case 14:
//     contours[index] = compute::Segment{.start = left, .end = bottom};
//     break;

//   case 2:
//   case 13:
//     contours[index] = compute::Segment{.start = bottom, .end = right};
//     break;

//   case 3:
//   case 12:
//     contours[index] = compute::Segment{.start = left, .end = right};
//     break;

//   case 4:
//   case 11:
//     contours[index] = compute::Segment{.start = top, .end = right};
//     break;

//   case 6:
//   case 9:
//     contours[index] = compute::Segment{.start = top, .end = bottom};
//     break;

//   case 7:
//   case 8:
//     contours[index] = compute::Segment{.start = left, .end = top};
//     break;

//   default:
//     break;
//   }
// }
// __global__ void marching_squares_part_2_double(compute::Grid heights, compute::Segment* contours, int2* tmp_2,
//                                                int count_1, int count_2, float threshold) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index > count_2) {
//     return;
//   }

//   int2 value = tmp_2[index];

//   int col = value.x;
//   int row = value.y;

//   float top_left = heights[row][col];
//   float top_right = heights[row][col + 1];
//   float bottom_right = heights[row + 1][col + 1];
//   float bottom_left = heights[row + 1][col];

//   bool inside = compute_inside(top_left, top_right, bottom_right, bottom_left, threshold);
//   int type = compute_type(top_left, top_right, bottom_right, bottom_left, threshold);

//   Point top{.x = col + linear_interpolation_factor(top_left, top_right, threshold), .y = (float)row};
//   Point right{.x = (float)(col + 1), .y = row + linear_interpolation_factor(top_right, bottom_right, threshold)};
//   Point bottom{.x = col + linear_interpolation_factor(bottom_left, bottom_right, threshold), .y = (float)(row + 1)};
//   Point left{.x = (float)col, .y = row + linear_interpolation_factor(top_left, bottom_left, threshold)};

//   int pos = count_1 + 2 * index;

//   switch (type) {
//   case 5:
//     if (inside) {
//       contours[pos] = compute::Segment{.start = left, .end = top};
//       contours[pos + 1] = compute::Segment{.start = bottom, .end = right};
//     } else {
//       contours[pos] = compute::Segment{.start = left, .end = bottom};
//       contours[pos + 1] = compute::Segment{.start = top, .end = right};
//     }
//     break;

//   case 10:
//     if (inside) {
//       contours[pos] = compute::Segment{.start = top, .end = right};
//       contours[pos + 1] = compute::Segment{.start = left, .end = bottom};
//     } else {
//       contours[pos] = compute::Segment{.start = top, .end = left};
//       contours[pos + 1] = compute::Segment{.start = right, .end = bottom};
//     }
//     break;

//   default:
//     break;
//   }
// }

} // namespace perf