#pragma once

#include "utils/grid.h"
#include "utils/math.h"

#include "compute/types.cuh"

namespace perf {

__global__ void marching_squares_part_1(compute::CGrid heights, int* __restrict__ count, int max_count,
                                        int2* __restrict__ tmp_coordinates, float const* __restrict__ thresholds,
                                        float* __restrict__ tmp_thresholds, int threshold_count);
__global__ void marching_squares_part_2(compute::CGrid heights, float4* __restrict__ contours, int count,
                                        int2 const* __restrict__ tmp_coordinates,
                                        float const* __restrict__ tmp_thresholds);
} // namespace perf