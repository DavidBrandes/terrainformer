#include "graphics/scene.h"

Scene::Scene(Config const& config, HeightGrid const& height_grid)
    : contour(ContourLayer{config, height_grid}), map(MapLayer{height_grid}), circle(CircleLayer{}) {}

std::shared_ptr<Scene> make_scene(Config const& config, HeightGrid const& height_grid) {
  return std::make_shared<Scene>(config, height_grid);
}