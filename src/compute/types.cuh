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

struct Segment {
  Point start;
  Point end;
};

struct Segments {
  Segment* segments;
  int* count;
  int maxCount;

  __device__ Segment& operator[](int index) { return segments[index]; }
};

struct Grid {
  float* __restrict__ values;
  Size size;

  __device__ float* operator[](int row) { return values + row * size.width; }
};

struct CGrid {
  float const* __restrict__ values;
  Size size;

  __device__ float const* operator[](int row) const { return values + row * size.width; }
};

struct ConstrainedGrid : public Grid {
  ConstrainedGrid(Grid grid, Range range_) : Grid(grid), range(range_) {}
  ConstrainedGrid() = default;

  Range range;
};

} // namespace compute