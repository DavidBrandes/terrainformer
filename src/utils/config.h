#pragma once

#include "utils/geometry.h"

struct WindowConfig {
  Size size;
  bool fullscreen;
  bool headless = false;
};

struct GridConfig {
  Size size;
};

struct MouseConfig {
  int scrollSteps;
};

struct ToolConfig {
  Range effectRadius;           // in grid units; limited by max(grid.width, grid.height)
  float heightBrushSensitivity; // in the range [0, 1]; 0 has no and 1 max effect
};

struct SceneConfig {
  bool showVertices;
  bool preserveAspectRatio;
};

struct Config {
  WindowConfig window;
  GridConfig grid;
  MouseConfig mouse;
  ToolConfig tool;
  SceneConfig scene;
};

Config load_config();
