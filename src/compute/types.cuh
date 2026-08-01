#pragma once

#include "utils/geometry.h"
#include "utils/grid.h"
#include "utils/math.h"

namespace compute {

struct Segments {
  float4* __restrict__ values;
  int* __restrict__ count;
  int maxCount;

  __device__ float4& operator[](int index) { return values[index]; }
};

struct Thresholds {
  float const* __restrict__ values;
  int count;
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