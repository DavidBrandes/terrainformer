#pragma once

#include "compute/parameters.h"
#include "utils/geometry.h"

__global__ void marching_squares(float* heights, float* contours, Size grid_size, float threshold);

__global__ void smoothstep(float* heights, Size grid_size, Range height_range, BrushDab brush_dab);
