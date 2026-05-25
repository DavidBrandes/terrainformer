#pragma once

#include "graphics/layers/circle.h"
#include "graphics/layers/contour.h"
#include "graphics/layers/map.h"
#include "utils/height_grid.h"

#include <memory>

struct Scene {
  Scene(Config const& config, HeightGrid const& height_grid);

  ContourLayer contour;
  MapLayer map;
  CircleLayer circle;
};

std::shared_ptr<Scene> make_scene(Config const& config, HeightGrid const& height_grid);