#include "utils/math.h"

#include <stdexcept>

std::vector<float> linspace(Range range, int steps, Bounds bounds) {
  if (steps == 0) {
    return {};
  }

  if (steps == 1) {
    return {range.midpoint()};
  }

  std::vector<float> values;

  int offset, divisor;
  switch (bounds) {
  case Bounds::INCLUDE:
    offset = 0;
    divisor = steps - 1;
    break;
  case Bounds::EXCLUDE:
    offset = 1;
    divisor = steps + 1;
    break;
  default:
    throw std::logic_error("Unknown Bounds value");
  }

  for (int i = 0; i < steps; ++i) {
    float t = static_cast<float>(i + offset) / static_cast<float>(divisor);
    float value = range.min + t * range.span();
    values.push_back(value);
  }

  return values;
}