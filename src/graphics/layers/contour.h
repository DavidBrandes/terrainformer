#pragma once

#include "utils/height_grid.h"

#include <glad/gl.h>

class ContourLayer {
public:
  ContourLayer(HeightGrid const& height_grid);
  ~ContourLayer();

  ContourLayer(ContourLayer const&) = delete;
  ContourLayer& operator=(ContourLayer const&) = delete;

  ContourLayer(ContourLayer&& other) = delete;
  ContourLayer& operator=(ContourLayer&& other) = delete;

  GLuint vao() const { return _vao; }
  GLuint vbo() const { return _vbo; }

  GLsizei segmentCount() const { return _segmentCount; }

private:
  GLuint _vao;
  GLuint _vbo;
  GLsizei _segmentCount;
};