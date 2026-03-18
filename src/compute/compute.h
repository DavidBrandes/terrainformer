#pragma once

#include "compute/parameters.h"
#include "compute/resources.h"
#include "parameters.h"
#include "utils/geometry.h"

Region brush_dab_region(BrushDab brush_dab, Size size);

void compute_contour(std::unique_ptr<MappedGpuResources> resources, Size grid_size, float threshold);

void modify_height(std::unique_ptr<MappedGpuResources> resources, Size grid_size, Range height_range,
                   BrushDab brush_dab, float contour_threshold);