#pragma once

#include "utils/config.h"
#include "utils/geometry.h"

#include <vector>

struct HeightGrid {
  static constexpr Range HEIGHT_RANGE{.min = -1.0f, .max = 1.0f};

  Size size;

  std::vector<float> positions;
  std::vector<float> heights;
};

HeightGrid make_sinusoidal_height_grid(GridConfig const& config);
