#pragma once

#include "utils/geometry.h"

struct BrushDab {
  Circle circle;
  float intensity;
};

struct Buffers {
  float* src;
  float* dst;
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