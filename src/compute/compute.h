#pragma once

#include "compute/resources.h"
#include "utils/types.h"

namespace compute {

class MappedGpuResources;

struct Result {
  int contourSegmentCount;
};

Result compute_contour(std::shared_ptr<MappedGpuResources> resources, float threshold);

Result modify_height(std::shared_ptr<MappedGpuResources> resources, Range height_range, BrushDab brush_dab,
                     float contour_threshold);

} // namespace compute