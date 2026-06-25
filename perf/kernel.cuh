#pragma once

#include "utils/types.h"

#include "compute/types.cuh"

namespace perf {

// struct Counts {
//   int count_1;
//   int count_2;
// };

// __global__ void marching_squares(compute::Grid heights, compute::Segment* contours, int* count, int max_count,
//                                  float threshold);
__global__ void marching_squares_part_1(compute::Grid heights, int* count, int max_count, int2* tmp, float threshold);
__global__ void marching_squares_part_2(compute::Grid heights, compute::Segment* contours, int count, int2* tmp,
                                        float threshold);
// __global__ void marching_squares_part_2(compute::Grid heights, compute::Segment* contours, int* count, int2* tmp,
//                                         float threshold);
// __global__ void marching_squares_part_1_split(compute::Grid heights, int2* tmp_1, int2* tmp_2, Counts* counts,
//                                               float threshold);
// __global__ void marching_squares_part_2_single(compute::Grid heights, compute::Segment* contours, int2* tmp_1,
//                                                int count_1, float threshold);
// __global__ void marching_squares_part_2_double(compute::Grid heights, compute::Segment* contours, int2* tmp_2,
//                                                int count_1, int count_2, float threshold);

void double_launch(compute::Grid heights, compute::Segments contours, int2* tmp, float threshold);
} // namespace perf