#include "config.h"

#include <toml++/toml.h>

Config load_config() {
  auto tbl = toml::parse_file(CONFIG_FILE_PATH);

  Config config;

  config.window.size.width = tbl["window"]["width"].value<int>().value();
  config.window.size.height = tbl["window"]["height"].value<int>().value();
  config.window.fullscreen = tbl["window"]["fullscreen"].value<bool>().value();

  config.grid.size.width = tbl["grid"]["width"].value<int>().value();
  config.grid.size.height = tbl["grid"]["height"].value<int>().value();

  config.mouse.scrollSteps = tbl["mouse"]["scroll_steps"].value<int>().value();

  config.tool.effectRadius.min = tbl["tool"]["min_effect_radius"].value<float>().value();
  config.tool.effectRadius.max = tbl["tool"]["max_effect_radius"].value<float>().value();
  config.tool.heightBrushSensitivity = tbl["tool"]["height_brush_sensitivity"].value<float>().value();

  config.scene.showVertices = tbl["scene"]["show_vertices"].value<bool>().value();
  config.scene.preserveAspectRatio = tbl["scene"]["preserve_aspect_ratio"].value<bool>().value();

  return config;
}
