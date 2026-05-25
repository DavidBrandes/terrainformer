#include "config.h"

#include <toml++/toml.h>

#include <string>

namespace {
GridConfig make_grid_config(toml::parse_result const& tbl) {
  GridConfig config;

  config.size.width = tbl["grid"]["size"]["width"].value<int>().value();
  config.size.height = tbl["grid"]["size"]["height"].value<int>().value();

  std::string type = tbl["grid"]["initialization"]["type"].value<std::string>().value();

  if (type == "perlin") {
    GridConfig::PerlinNoiseConfig perlin_config;

    perlin_config.seed = tbl["grid"]["initialization"]["perlin"]["seed"].value<std::uint32_t>().value();
    perlin_config.frequency = tbl["grid"]["initialization"]["perlin"]["frequency"].value<float>().value();
    perlin_config.octaves = tbl["grid"]["initialization"]["perlin"]["octaves"].value<int>().value();
    perlin_config.persistence = tbl["grid"]["initialization"]["perlin"]["persistence"].value<float>().value();

    config.initialization = perlin_config;

  } else if (type == "sinusoidal") {
    GridConfig::SinusoidalConfig sinusoidal_config;

    sinusoidal_config.amplitude = tbl["grid"]["initialization"]["sinusoidal"]["amplitude"].value<float>().value();
    sinusoidal_config.frequency = tbl["grid"]["initialization"]["sinusoidal"]["frequency"].value<float>().value();

    config.initialization = sinusoidal_config;

  } else {
    throw std::runtime_error("[make_grid_config]: Unknown terrain initialization type: " + type);
  }

  return config;
}

} // namespace

Config load_config() {
  toml::parse_result tbl = toml::parse_file(CONFIG_FILE_PATH);

  Config config;

  config.window.size.width = tbl["window"]["width"].value<int>().value();
  config.window.size.height = tbl["window"]["height"].value<int>().value();
  config.window.fullscreen = tbl["window"]["fullscreen"].value<bool>().value();

  config.grid = make_grid_config(tbl);

  config.tool.effectRadius.min = tbl["tool"]["min_effect_radius"].value<float>().value();
  config.tool.effectRadius.max = tbl["tool"]["max_effect_radius"].value<float>().value();
  config.tool.heightBrushSensitivity = tbl["tool"]["height_brush_sensitivity"].value<float>().value();
  config.tool.scrollSteps = tbl["tool"]["scroll_steps"].value<int>().value();

  config.scene.preserveAspectRatio = tbl["scene"]["preserve_aspect_ratio"].value<bool>().value();
  config.scene.contourCount = tbl["scene"]["contour_count"].value<int>().value();
  config.scene.maxContourSegmentFraction = tbl["scene"]["max_contour_segment_fraction"].value<float>().value();

  return config;
}
