#pragma once

#include "utils/types.h"

#include <glad/gl.h>

class CircleLayer {
public:
  static constexpr GLsizei CIRCLE_SEGMENTS = 64;

  CircleLayer();
  ~CircleLayer();

  CircleLayer(CircleLayer const&) = delete;
  CircleLayer& operator=(CircleLayer const&) = delete;

  CircleLayer(CircleLayer&& other) = delete;
  CircleLayer& operator=(CircleLayer&& other) = delete;

  GLuint vao() const { return _vao; }

  struct Parameters {
    Circle circle;
    bool visible;
  };

  Parameters parameters() const { return _parameters; }
  void update(Parameters parameters) { _parameters = parameters; }

private:
  GLuint _vao;
  GLuint _vbo;

  Parameters _parameters{};
};
