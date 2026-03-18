#include "common/setup.h"

#include "compute/parameters.h"

#include <algorithm>

namespace perf {

constexpr float BRUSH_INTENSITY = 0.01f;

Point point_from_normalized(float x_normalized, float y_normalized) {
  float x = x_normalized * (GRID_CONFIG.size.width - 1);
  float y = y_normalized * (GRID_CONFIG.size.height - 1);

  return Point{x, y};
}

float radius_from_normalized(float radius_normalized) {
  // TODO check if compiler precomputes the min
  return (std::min(GRID_CONFIG.size.height, GRID_CONFIG.size.width) - 1) * radius_normalized;
}

BrushDab make_brush_dab(Point center, float radius) {
  Circle circle{.center = center, .radius = radius};

  return BrushDab{.circle = circle, .intensity = BRUSH_INTENSITY};
}

MappedHeightGrid make_height_grid() {
  HeightGrid height_grid = make_sinusoidal_height_grid(GRID_CONFIG);

  return MappedHeightGrid{std::move(height_grid)};
}

std::vector<BrushDab> generate_brush_dabs(int steps) {
  std::vector<BrushDab> result;

  if (steps < 1) {
    return result;
  }

  if (steps == 1) {
    Point center = point_from_normalized(0.5f, 0.5f);
    float radius = radius_from_normalized(0.5f);
    BrushDab brush_dab = make_brush_dab(center, radius);

    result.push_back(brush_dab);
    return result;
  }

  for (int i_height = 0; i_height < steps; ++i_height) {
    for (int i_width = 0; i_width < steps; ++i_width) {
      for (int i = 1; i <= steps; ++i) {
        float x_normalized = static_cast<float>(i_width) / static_cast<float>(steps - 1);
        float y_normalized = static_cast<float>(i_height) / static_cast<float>(steps - 1);
        float radius_normalized = static_cast<float>(i) / static_cast<float>(steps);

        Point center = point_from_normalized(x_normalized, y_normalized);
        float radius = radius_from_normalized(radius_normalized);
        BrushDab brush_dab = make_brush_dab(center, radius);

        result.push_back(brush_dab);
      }
    }
  }

  return result;
}

} // namespace perf
