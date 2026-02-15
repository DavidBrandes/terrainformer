#include "utils/height_grid.h"

#include "utils/config.h"

#include <cmath>

HeightGrid make_sinusoidal_height_grid(Config const& config) {
  std::vector<float> positions;
  std::vector<float> heights;

  heights.reserve(static_cast<size_t>(config.grid.size.width * config.grid.size.height));
  positions.reserve(static_cast<size_t>(config.grid.size.width * config.grid.size.height * 2));

  float offset = HeightGrid::HEIGHT_RANGE.midpoint();
  float scale = HeightGrid::HEIGHT_RANGE.span() / 4; // filling half of the range

  for (int i = 0; i < config.grid.size.height; ++i) {
    for (int j = 0; j < config.grid.size.width; ++j) {
      float x = static_cast<float>(j);
      float y = static_cast<float>(i);

      float x_norm = (x / static_cast<float>(config.grid.size.width - 1)) * 2.0f - 1.0f;
      float y_norm = (y / static_cast<float>(config.grid.size.height - 1)) * 2.0f - 1.0f;

      float height = offset + scale * sinf(x_norm * PI) * cosf(y_norm * PI);

      positions.push_back(x);
      positions.push_back(y);
      heights.push_back(height);
    }
  }

  return HeightGrid{.size = config.grid.size, .positions = positions, .heights = heights};
}
