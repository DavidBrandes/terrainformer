#pragma once

#include "utils/height_grid.h"

#include <glad/gl.h>

class MapLayer {
public:
  MapLayer(HeightGrid const&);
  ~MapLayer();

  MapLayer(MapLayer const&) = delete;
  MapLayer& operator=(MapLayer const&) = delete;

  MapLayer(MapLayer&& other) = delete;
  MapLayer& operator=(MapLayer&& other) = delete;

  GLuint vao() const { return _vao; }
  GLuint heightsVbo() const { return _heightsVbo; }
  GLsizei indexCount() const { return _indexCount; }
  Size size() const { return _size; }
  Range range() const { return _range; }

private:
  GLuint _vao;
  GLuint _positionsVbo;
  GLuint _heightsVbo;
  GLuint _ebo;
  GLsizei _indexCount;

  Size _size;
  Range _range;
};