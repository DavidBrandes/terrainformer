#pragma once
#include "utils/geometry.h"

struct Size {
  int width;
  int height;

  bool empty() const { return width <= 0 || height <= 0; }
};

struct Vertex {
  int col;
  int row;
};

struct Region {
  Vertex origin;
  Size size;

  bool empty() const { return size.empty(); }
};

struct BrushDab {
  Circle circle;
  float intensity;
};