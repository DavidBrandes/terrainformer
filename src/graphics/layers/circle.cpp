#include "graphics/layers/circle.h"

#include <cmath>
#include <vector>

namespace {

std::vector<float> generate_unit_circle() {
  std::vector<float> vertices;
  vertices.reserve(CircleLayer::CIRCLE_SEGMENTS * 2); // 1 vertex per segment, 2 floats per vertex

  for (int i = 0; i < CircleLayer::CIRCLE_SEGMENTS; ++i) {
    float angle = (static_cast<float>(i) / static_cast<float>(CircleLayer::CIRCLE_SEGMENTS)) * 2.0f * PI;
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    vertices.push_back(cos_a);
    vertices.push_back(sin_a);
  }

  return vertices;
}

} // namespace

CircleLayer::CircleLayer() {
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);

  std::vector<float> circle_vertices = generate_unit_circle();

  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(circle_vertices.size() * sizeof(float)), circle_vertices.data(),
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

CircleLayer::~CircleLayer() {
  glDeleteBuffers(1, &_vbo);
  glDeleteVertexArrays(1, &_vao);
}
