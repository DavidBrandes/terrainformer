#include "graphics/layers/contour.h"

#include "utils/height_grid.h"

#include <glad/gl.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>

namespace {
std::vector<float> compute_gradients(int contour_count) {
  std::vector<float> gradients;

  if (contour_count < 1) {
    return gradients;
  }

  if (contour_count == 1) {
    gradients.push_back(0.5f);
    return gradients;
  }

  for (int i = 0; i < contour_count; ++i) {
    float gradient = static_cast<float>(i) / static_cast<float>(contour_count - 1);
    gradients.push_back(gradient);
  }

  return gradients;
}
} // namespace

ContourLayer::ContourLayer(Config const& config, HeightGrid const& height_grid) {
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);

  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);

  int absolute_max_segments =
      (height_grid.size.width - 1) * (height_grid.size.height - 1) * 2 * config.scene.contourCount;
  int computed_segments;
  if (config.scene.maxContourSegmentFraction < 1.0f) {
    computed_segments = static_cast<int>(
        std::round(config.scene.maxContourSegmentFraction * static_cast<float>(absolute_max_segments)));
  } else {
    computed_segments = absolute_max_segments;
  }
  int max_segments = std::clamp(computed_segments, 0, absolute_max_segments);
  _maxSegments = max_segments;

  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<size_t>(max_segments * 4) * sizeof(float)), nullptr,
               GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  _offsets = std::vector<GLsizei>(static_cast<size_t>(config.scene.contourCount), 0);
  _gradients = compute_gradients(config.scene.contourCount);
  _count = config.scene.contourCount;
}

ContourLayer::~ContourLayer() {
  glDeleteBuffers(1, &_vbo);
  glDeleteVertexArrays(1, &_vao);
}

void ContourLayer::update(std::vector<int>&& offsets) {
  static_assert(std::same_as<int, GLsizei>);
  if (offsets.size() != _offsets.size()) {
    throw std::runtime_error(
        std::format("[ContourLayer::update] Cannot update offsets of size {} with a vector of size {}", _offsets.size(),
                    offsets.size()));
  }

  _offsets = offsets;
}