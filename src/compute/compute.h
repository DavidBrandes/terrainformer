#pragma once

#include "compute/parameters.h"
#include "compute/resources.h"
#include "utils/geometry.h"

void compute_contour(std::unique_ptr<MappedGpuResources> resources, Size grid_size, float threshold);

void modify_height(std::unique_ptr<MappedGpuResources> resources, Size grid_size, Range height_range,
                   BrushDab brush_dab, float contour_threshold);