#pragma once

#include "compute/resources.h"
#include "utils/geometry.h"
#include "utils/grid.h"

namespace compute {

class MappedGpuResources;

struct Result {
  bool modified;
  int contourSegmentCount;
};

Result compute_contour(std::shared_ptr<MappedGpuResources> resources);

Result modify_height(std::shared_ptr<MappedGpuResources> resources, BrushDab brush_dab);

} // namespace compute