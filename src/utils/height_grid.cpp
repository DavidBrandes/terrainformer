#include "utils/height_grid.h"

#include "utils/config.h"

#include <algorithm>
#include <cmath>

HeightGrid make_sinusoidal_height_grid(GridConfig const& config, float frequency) {
  std::vector<float> positions;
  std::vector<float> heights;

  heights.reserve(static_cast<size_t>(config.size.width * config.size.height));
  positions.reserve(static_cast<size_t>(config.size.width * config.size.height * 2));

  float offset = HeightGrid::HEIGHT_RANGE.midpoint();
  float scale = HeightGrid::HEIGHT_RANGE.span() / 4; // filling half of the range
  float normalization_factor =
      std::min(static_cast<float>(config.size.width), static_cast<float>(config.size.height)) - 1;

  for (int i = 0; i < config.size.height; ++i) {
    for (int j = 0; j < config.size.width; ++j) {
      float x = static_cast<float>(j);
      float y = static_cast<float>(i);

      float x_norm = (x / normalization_factor);
      float y_norm = (y / normalization_factor);

      float height = offset + scale * sinf(x_norm * PI * frequency) * cosf(y_norm * PI * frequency);

      positions.push_back(x);
      positions.push_back(y);
      heights.push_back(height);
    }
  }

  return HeightGrid{.size = config.size, .positions = positions, .heights = heights};
}
