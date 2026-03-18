#pragma once

#include "compute/parameters.h"
#include "utils/config.h"
#include "utils/height_grid.h"

#include <functional>

namespace perf {

constexpr GridConfig GRID_CONFIG{.size = Size{.width = 1920, .height = 1080}};

struct MappedHeightGrid : public HeightGrid {
  MappedHeightGrid(HeightGrid height_grid);
  ~MappedHeightGrid();

  MappedHeightGrid(MappedHeightGrid const&) = delete;
  MappedHeightGrid& operator=(MappedHeightGrid const&) = delete;

  MappedHeightGrid(MappedHeightGrid&&) = delete;
  MappedHeightGrid& operator=(MappedHeightGrid&&) = delete;

  float* mappedHeights;
};

struct ScratchBuffer {
  ScratchBuffer(int size);
  ~ScratchBuffer();

  ScratchBuffer(ScratchBuffer const&) = delete;
  ScratchBuffer& operator=(ScratchBuffer const&) = delete;

  ScratchBuffer(ScratchBuffer&&) = delete;
  ScratchBuffer& operator=(ScratchBuffer&&) = delete;

  float* ptr;
};

float benchmark(std::function<void()> func, int iterations, int warmup_iterations);

void launch_smoothstep(MappedHeightGrid const& height_grid, ScratchBuffer const& scratch_buffer, BrushDab brush_dab);

} // namespace perf
