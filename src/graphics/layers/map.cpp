#include "graphics/layers/map.h"

#include <glad/gl.h>

#include <vector>

namespace {

std::vector<unsigned int> generate_grid_indices(HeightGrid const& height_grid) {
  std::vector<unsigned int> indices;
  indices.reserve(static_cast<size_t>((height_grid.size.height - 1) * (height_grid.size.width - 1) * 6));

  for (int i = 0; i < height_grid.size.height - 1; ++i) {
    for (int j = 0; j < height_grid.size.width - 1; ++j) {
      unsigned int top_left = static_cast<unsigned int>(i * height_grid.size.width + j);
      unsigned int top_right = top_left + 1;
      unsigned int bottom_left = static_cast<unsigned int>((i + 1) * height_grid.size.width + j);
      unsigned int bottom_right = bottom_left + 1;

      // First triangle
      indices.push_back(top_left);
      indices.push_back(bottom_left);
      indices.push_back(top_right);

      // Second triangle
      indices.push_back(top_right);
      indices.push_back(bottom_left);
      indices.push_back(bottom_right);
    }
  }

  return indices;
}

} // namespace

MapLayer::MapLayer(HeightGrid const& height_grid) {
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_positionsVbo);
  glGenBuffers(1, &_heightsVbo);
  glGenBuffers(1, &_ebo);

  glBindVertexArray(_vao);

  glBindBuffer(GL_ARRAY_BUFFER, _positionsVbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(height_grid.positions.size() * sizeof(float)),
               height_grid.positions.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, _heightsVbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(height_grid.heights.size() * sizeof(float)),
               height_grid.heights.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);

  std::vector<unsigned int> indices = generate_grid_indices(height_grid);
  _indexCount = static_cast<GLsizei>(indices.size());

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)), indices.data(),
               GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

MapLayer::~MapLayer() {
  glDeleteBuffers(1, &_positionsVbo);
  glDeleteBuffers(1, &_heightsVbo);
  glDeleteBuffers(1, &_ebo);
  glDeleteVertexArrays(1, &_vao);
}

