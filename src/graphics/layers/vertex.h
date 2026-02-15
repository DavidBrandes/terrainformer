#pragma once

#include "utils/height_grid.h"

#include <glad/gl.h>

class VertexLayer {
public:
  VertexLayer(HeightGrid const& height_grid);
  ~VertexLayer();

  VertexLayer(VertexLayer const&) = delete;
  VertexLayer& operator=(VertexLayer const&) = delete;

  VertexLayer(VertexLayer&& other) = delete;
  VertexLayer& operator=(VertexLayer&& other) = delete;

  void update(bool visible) { _visible = visible; }

  GLuint vao() const { return _vao; }
  GLsizei vertexCount() const { return _vertexCount; }
  bool visible() const { return _visible; }

private:
  GLuint _vao;
  GLuint _positionsVbo;
  GLsizei _vertexCount;
  bool _visible;
};
