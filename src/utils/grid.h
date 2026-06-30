#pragma once
#include "utils/geometry.h"

struct Size {
  int width;
  int height;

  bool empty() const { return width <= 0 || height <= 0; }
};

struct BrushDab {
  Circle circle;
  float intensity;
};