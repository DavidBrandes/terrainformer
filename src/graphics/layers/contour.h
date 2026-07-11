#pragma once

#include "utils/height_grid.h"

#include <glad/gl.h>

class ContourLayer {
public:
  ContourLayer(Config const& config, HeightGrid const& height_grid);
  ~ContourLayer();

  ContourLayer(ContourLayer const&) = delete;
  ContourLayer& operator=(ContourLayer const&) = delete;

  ContourLayer(ContourLayer&& other) = delete;
  ContourLayer& operator=(ContourLayer&& other) = delete;

  GLuint vao() const { return _vao; }
  GLuint vbo() const { return _vbo; }

  void update(int count);

  int count() const { return _count; }
  int maxCount() const { return _maxCount; }

private:
  GLuint _vao;
  GLuint _vbo;
  int _count;
  int _maxCount;
};