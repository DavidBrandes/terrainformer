#include "graphics/layers/contour.h"

#include "utils/height_grid.h"

#include <glad/gl.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
int compute_max_segments(Config const& config, HeightGrid const& height_grid) {
  int grid_elements = (height_grid.size.width - 1) * (height_grid.size.height - 1);
  int absolute_max_segments = grid_elements * 2 * config.scene.contourCount; // Max two segments per element

  int computed_segments;
  if (config.scene.maxContourSegmentFraction < 1.0f) {
    float fractional_segments = config.scene.maxContourSegmentFraction * static_cast<float>(absolute_max_segments);
    computed_segments = static_cast<int>(std::round(fractional_segments));

  } else {
    computed_segments = absolute_max_segments;
  }

  return std::clamp(computed_segments, 0, absolute_max_segments);
}
} // namespace

ContourLayer::ContourLayer(Config const& config, HeightGrid const& height_grid) {
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);

  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);

  _maxCount = compute_max_segments(config, height_grid);

  // Each segment consists out of 2 (x, y) points, i.e. 4 elements
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<size_t>(_maxCount * 4) * sizeof(float)), nullptr,
               GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  _count = 0;
}

ContourLayer::~ContourLayer() {
  glDeleteBuffers(1, &_vbo);
  glDeleteVertexArrays(1, &_vao);
}

void ContourLayer::update(int count) { _count = std::min(count, _maxCount); }