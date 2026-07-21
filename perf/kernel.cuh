#pragma once

#include "utils/grid.h"
#include "utils/math.h"

#include "compute/types.cuh"

namespace perf {

__global__ void marching_squares(compute::CGrid heights, int* __restrict__ count, int max_count,
                                 float const* __restrict__ thresholds, float4* __restrict__ contours,
                                 int threshold_count);
} // namespace perf