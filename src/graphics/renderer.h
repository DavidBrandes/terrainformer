#pragma once

#include "app/state.h"
#include "graphics/scene.h"
#include "utils/config.h"
#include "utils/types.h"

#include <glad/gl.h>

class Renderer {
public:
  Renderer();
  ~Renderer();

  Renderer(Renderer const&) = delete;
  Renderer& operator=(Renderer const&) = delete;

  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;

  void render(std::shared_ptr<Scene> scene, ApplicationState const& state);

private:
  void clear();
  void initializeShaders();
  void setCropRegion(ApplicationState const& state);
  void renderContours(ContourLayer const& contours);
  void renderMap(MapLayer const& map);
  void renderCircle(CircleLayer const& circle);
  void renderVertices(VertexLayer const& vertices);

  GLuint _contourShaderProgram;
  GLuint _mapShaderProgram;
  GLuint _circleShaderProgram;
  GLuint _vertexShaderProgram;

  static constexpr char const* CONTOUR_VERT_SHADER_PATH = SHADERS_DIR "/contour.vert";
  static constexpr char const* CONTOUR_FRAG_SHADER_PATH = SHADERS_DIR "/contour.frag";
  static constexpr char const* MAP_VERT_SHADER_PATH = SHADERS_DIR "/map.vert";
  static constexpr char const* MAP_FRAG_SHADER_PATH = SHADERS_DIR "/map.frag";
  static constexpr char const* CIRCLE_VERT_SHADER_PATH = SHADERS_DIR "/circle.vert";
  static constexpr char const* CIRCLE_FRAG_SHADER_PATH = SHADERS_DIR "/circle.frag";
  static constexpr char const* VERTEX_VERT_SHADER_PATH = SHADERS_DIR "/vertex.vert";
  static constexpr char const* VERTEX_FRAG_SHADER_PATH = SHADERS_DIR "/vertex.frag";

  static constexpr float CIRCLE_LINE_WIDTH = 2.0f;
  static constexpr float CONTOUR_LINE_WIDTH = 1.5f;
  static constexpr float VERTEX_POINT_SIZE = 5.0f;
};