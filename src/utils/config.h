#pragma once

#include "utils/grid.h"
#include "utils/math.h"

#include <cstdint>
#include <variant>

struct WindowConfig {
  Size size;
  bool fullscreen;
  bool headless = false;
};

struct GridConfig {
  Size size;

  struct PerlinNoiseConfig {
    std::uint32_t seed;
    float frequency;
    int octaves;
    float persistence;
  };

  struct SinusoidalConfig {
    float amplitude; // clamped to 1
    float frequency;
  };

  using InitializationConfig = std::variant<PerlinNoiseConfig, SinusoidalConfig>;

  InitializationConfig initialization;
};

struct ToolConfig {
  Range effectRadius;           // in grid units; limited by max(grid.width, grid.height)
  float heightBrushSensitivity; // scaled to the range
  int scrollSteps;
};

struct SceneConfig {
  bool preserveAspectRatio;
  int contourCount;
  float maxContourSegmentFraction; // max segments to display as fraction of absolute max
};

struct Config {
  WindowConfig window;
  GridConfig grid;
  ToolConfig tool;
  SceneConfig scene;
};

Config load_config();
