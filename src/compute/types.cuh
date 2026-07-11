#pragma once

#include "utils/geometry.h"
#include "utils/grid.h"
#include "utils/math.h"

namespace compute {

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

struct ConstrainedGrid {
  Grid grid;
  Range range;
};

} // namespace compute