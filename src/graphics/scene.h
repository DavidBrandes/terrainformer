#pragma once

#include "graphics/layers/circle.h"
#include "graphics/layers/contour.h"
#include "graphics/layers/map.h"
#include "graphics/layers/vertex.h"
#include "utils/height_grid.h"

#include <memory>

struct Scene {
  Scene(HeightGrid const& height_grid);

  ContourLayer contour;
  MapLayer map;
  CircleLayer circle;
  VertexLayer vertex;
};

std::shared_ptr<Scene> make_scene(HeightGrid const& height_grid);