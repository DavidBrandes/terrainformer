#pragma once

#include "utils/types.h"

namespace compute {

struct Vertex {
  int col;
  int row;
};

struct Region {
  Vertex origin;
  Size size;

  bool empty() const { return size.empty(); }
};

struct Point {
  __device__ Point(float x_, float y_) : x(x_), y(y_) {}
  __device__ Point(float x_, int y_) : x(x_), y(static_cast<float>(y_)) {}
  __device__ Point(int x_, float y_) : x(static_cast<float>(x_)), y(y_) {}
  __device__ Point(int x_, int y_) : x(static_cast<float>(x_)), y(static_cast<float>(y_)) {}
  __device__ Point() = default;

  float x;
  float y;
};

struct Segment {
  __device__ Segment(Point start_, Point end_) : start(start_), end(end_) {}

  Point start;
  Point end;
};

struct Segments {
  Segment* segments;
  int* count;

  __device__ Segment& operator[](int index) { return segments[index]; }
};

struct Grid {
  float* values;
  Size size;

  __device__ float* operator[](int row) { return values + row * size.width; }
};

struct ConstrainedGrid : public Grid {
  ConstrainedGrid(Grid grid, Range range_) : Grid(grid), range(range_) {}
  ConstrainedGrid() = default;

  Range range;
};

} // namespace compute