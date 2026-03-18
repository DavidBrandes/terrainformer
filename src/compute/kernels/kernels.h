#pragma once

#include "compute/parameters.h"
#include "utils/geometry.h"

constexpr int SMOOTHSTEP_ROW_COARSE = 4;

__global__ void marching_squares(float* heights, float* contours, Size grid_size, float threshold);

__global__ void smoothstep(Buffers heights, Size grid_size, Vertex offset, Range height_range, BrushDab brush_dab);
