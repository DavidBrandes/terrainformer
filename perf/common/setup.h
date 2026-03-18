#pragma once

#include "common/device.h"
#include "compute/parameters.h"
#include "utils/geometry.h"

#include <vector>

namespace perf {

MappedHeightGrid make_height_grid();

Point point_from_normalized(float x_normalized, float y_normalized);
float radius_from_normalized(float radius_normalized);
BrushDab make_brush_dab(Point center, float radius);

std::vector<BrushDab> generate_brush_dabs(int steps);
} // namespace perf
