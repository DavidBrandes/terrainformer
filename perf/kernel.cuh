#pragma once

#include "utils/types.h"

#include "compute/types.cuh"

namespace perf {

constexpr int BLOCK_DIM = 256;
constexpr int WARP_SIZE = 32;

__global__ void marching_squares_part_1(compute::CGrid heights, int* __restrict__ count, int max_count,
                                        int2* __restrict__ tmp, float threshold);
__global__ void marching_squares_part_2(compute::CGrid heights, float4* __restrict__ contours, int count,
                                        int2 const* __restrict__ tmp, float threshold);

} // namespace perf