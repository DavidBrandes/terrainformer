#include "graphics/layers/vertex.h"

VertexLayer::VertexLayer(HeightGrid const& height_grid) {
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_positionsVbo);

  _vertexCount = static_cast<GLsizei>(height_grid.heights.size());

  glBindVertexArray(_vao);

  glBindBuffer(GL_ARRAY_BUFFER, _positionsVbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(height_grid.positions.size() * sizeof(float)),
               height_grid.positions.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

VertexLayer::~VertexLayer() {
  glDeleteBuffers(1, &_positionsVbo);
  glDeleteVertexArrays(1, &_vao);
}

