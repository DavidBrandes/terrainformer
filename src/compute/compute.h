#pragma once

#include "compute/resources.h"
#include "utils/types.h"

#include <vector>

namespace compute {

class MappedGpuResources;

struct Result {
  bool modified;
  std::vector<int> contourOffsets;
};

Result compute_contour(std::shared_ptr<MappedGpuResources> resources, std::vector<float> const& thresholds);

Result modify_height(std::shared_ptr<MappedGpuResources> resources, Range height_range, BrushDab brush_dab,
                     std::vector<float> const& thresholds);

} // namespace compute