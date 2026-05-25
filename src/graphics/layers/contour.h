#pragma once

#include "utils/height_grid.h"

#include <glad/gl.h>

#include <vector>

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

  void update(std::vector<int>&& offsets);

  std::vector<GLsizei> const& offsets() const { return _offsets; }
  std::vector<float> const& gradients() const { return _gradients; }
  int count() const { return _count; }
  int maxSegments() const { return _maxSegments; }

private:
  GLuint _vao;
  GLuint _vbo;
  std::vector<GLsizei> _offsets;
  std::vector<float> _gradients;
  int _count;
  int _maxSegments;
};