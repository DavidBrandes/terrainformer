#pragma once

#include <vector>

struct Range {
  float min;
  float max;

  constexpr float span() const { return max - min; }
  constexpr float midpoint() const { return (max + min) / 2.0f; }
};

enum class Bounds { INCLUDE, EXCLUDE };

std::vector<float> linspace(Range range, int steps, Bounds bounds);