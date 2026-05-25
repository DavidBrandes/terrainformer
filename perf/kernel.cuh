#pragma once

#include "utils/types.h"

#include "compute/types.cuh"

namespace perf {

__global__ void marching_squares(compute::Grid heights, compute::Segments contours, float threshold);
__global__ void marching_squares_types(compute::Grid heights, int* types, int* counter, float threshold);
__global__ void marching_squares_types_warp(compute::Grid heights, int* types, int* counter, float threshold);
__global__ void marching_squares_types_block(compute::Grid heights, int* types, int* counter, float threshold);
} // namespace perf