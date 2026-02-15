#include "compute/kernels/kernels.h"

#include <cmath>

__device__ float linear_interpolation_factor(float start, float end, float value) {
  if (fabsf(end - start) < 1e-6f) {
    return 0.5f;
  }

  return (value - start) / (end - start);
}

__global__ void marching_squares(float* heights, float* contours, Size grid_size, float threshold) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int n_vertices = (grid_size.height - 1) * (grid_size.width - 1);

  for (int i = index; i < n_vertices; i += stride) {
    int row = i / (grid_size.width - 1);
    int column = i % (grid_size.width - 1);

    float top_left = heights[row * grid_size.width + column];
    float top_right = heights[row * grid_size.width + (column + 1)];
    float bottom_right = heights[(row + 1) * grid_size.width + (column + 1)];
    float bottom_left = heights[(row + 1) * grid_size.width + column];

    int vertex_type = 0;

    if (top_left > threshold) {
      vertex_type |= 8;
    }
    if (top_right > threshold) {
      vertex_type |= 4;
    }
    if (bottom_right > threshold) {
      vertex_type |= 2;
    }
    if (bottom_left > threshold) {
      vertex_type |= 1;
    }

    // Edge crossing factors
    float top = column + linear_interpolation_factor(top_left, top_right, threshold);
    float right = row + linear_interpolation_factor(top_right, bottom_right, threshold);
    float bottom = column + linear_interpolation_factor(bottom_left, bottom_right, threshold);
    float left = row + linear_interpolation_factor(top_left, bottom_left, threshold);

    switch (vertex_type) {
    case 0:
    case 15:
      contours[2 * 4 * i] = nanf("");
      contours[2 * 4 * i + 1] = nanf("");
      contours[2 * 4 * i + 2] = nanf("");
      contours[2 * 4 * i + 3] = nanf("");

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");

      break;

    case 1:
    case 14:
      contours[2 * 4 * i] = column;
      contours[2 * 4 * i + 1] = left;
      contours[2 * 4 * i + 2] = bottom;
      contours[2 * 4 * i + 3] = row + 1;

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");
      break;

    case 2:
    case 13:
      contours[2 * 4 * i] = bottom;
      contours[2 * 4 * i + 1] = row + 1;
      contours[2 * 4 * i + 2] = column + 1;
      contours[2 * 4 * i + 3] = right;

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");
      break;

    case 3:
    case 12:
      contours[2 * 4 * i] = column;
      contours[2 * 4 * i + 1] = left;
      contours[2 * 4 * i + 2] = column + 1;
      contours[2 * 4 * i + 3] = right;

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");
      break;

    case 4:
    case 11:
      contours[2 * 4 * i] = top;
      contours[2 * 4 * i + 1] = row;
      contours[2 * 4 * i + 2] = column + 1;
      contours[2 * 4 * i + 3] = right;

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");
      break;

    case 5:
      contours[2 * 4 * i] = column;
      contours[2 * 4 * i + 1] = left;
      contours[2 * 4 * i + 2] = top;
      contours[2 * 4 * i + 3] = row;

      contours[2 * 4 * i + 4] = bottom;
      contours[2 * 4 * i + 5] = row + 1;
      contours[2 * 4 * i + 6] = column + 1;
      contours[2 * 4 * i + 7] = right;

      break;
    case 6:
    case 9:
      contours[2 * 4 * i] = top;
      contours[2 * 4 * i + 1] = row;
      contours[2 * 4 * i + 2] = bottom;
      contours[2 * 4 * i + 3] = row + 1;

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");
      break;

    case 7:
    case 8:
      contours[2 * 4 * i] = column;
      contours[2 * 4 * i + 1] = left;
      contours[2 * 4 * i + 2] = top;
      contours[2 * 4 * i + 3] = row;

      contours[2 * 4 * i + 4] = nanf("");
      contours[2 * 4 * i + 5] = nanf("");
      contours[2 * 4 * i + 6] = nanf("");
      contours[2 * 4 * i + 7] = nanf("");
      break;

    case 10:
      contours[2 * 4 * i] = top;
      contours[2 * 4 * i + 1] = row;
      contours[2 * 4 * i + 2] = column + 1;
      contours[2 * 4 * i + 3] = right;

      contours[2 * 4 * i + 4] = column;
      contours[2 * 4 * i + 5] = left;
      contours[2 * 4 * i + 6] = bottom;
      contours[2 * 4 * i + 7] = row + 1;

      break;

    default:
      break;
    }
  }
}