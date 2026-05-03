#include "graphics/layers/contour.h"

#include "utils/height_grid.h"

#include <glad/gl.h>

#include <cstddef>

ContourLayer::ContourLayer(HeightGrid const& height_grid) {
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);

  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);

  int max_segment_count = (height_grid.size.width - 1) * (height_grid.size.height - 1) * 2;
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<size_t>(max_segment_count * 4) * sizeof(float)),
               nullptr, GL_DYNAMIC_DRAW);
  _segmentCount = 0;

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

ContourLayer::~ContourLayer() {
  glDeleteBuffers(1, &_vbo);
  glDeleteVertexArrays(1, &_vao);
}

void ContourLayer::update(int segment_count) { _segmentCount = segment_count; }
