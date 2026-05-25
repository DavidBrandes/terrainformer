#pragma once

#include "utils/types.h"

struct MouseState {
  static constexpr Range SCROLL = {-1.0f, 1.0f};
  enum class Button { NONE, LEFT, RIGHT };

  Button button;
  Point pixel;
  float scroll; // in the range of SCROLL
};

struct WindowState {
  Size size;
};

struct CropState {
  Point min;
  Point max;
};

struct ToolState {
  enum class Tool { NONE, SHIFT };

  Tool activeTool = Tool::SHIFT;
};

struct SceneState {
  bool preserveAspectRatio;
};

struct ApplicationState {
  MouseState mouse;
  WindowState window;
  CropState crop;
  SceneState scene;
  ToolState tool;
};