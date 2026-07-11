#pragma once

#include "compute/resources.h"
#include "utils/geometry.h"
#include "utils/grid.h"

#include <vector>

namespace compute {

class MappedGpuResources;

struct Result {
  bool modified;
  int contourSegmentCount;
};

Result compute_contour(std::shared_ptr<MappedGpuResources> resources, std::vector<float> const& thresholds);

Result modify_height(std::shared_ptr<MappedGpuResources> resources, BrushDab brush_dab,
                     std::vector<float> const& thresholds);

} // namespace compute