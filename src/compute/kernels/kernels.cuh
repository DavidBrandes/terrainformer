#pragma once

#include "utils/grid.h"

#include "compute/types.cuh"

namespace compute {

__global__ void marching_squares(Grid heights, Segments contours, float threshold);

__global__ void smoothstep(ConstrainedGrid heights, Vertex offset, BrushDab brush_dab);

} // namespace compute