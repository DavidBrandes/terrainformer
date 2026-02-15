#include "graphics/scene.h"

Scene::Scene(HeightGrid const& height_grid)
    : contour(ContourLayer{height_grid}), map(MapLayer{height_grid}), circle(CircleLayer{}),
      vertex(VertexLayer{height_grid}) {}

std::shared_ptr<Scene> make_scene(HeightGrid const& height_grid) { return std::make_shared<Scene>(height_grid); }