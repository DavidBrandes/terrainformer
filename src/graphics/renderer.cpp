#include "graphics/renderer.h"

#include "graphics/layers/contour.h"

#include <cstddef>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string load_shader_source(char const* filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error(std::format("[load_shader_source]: Failed to open shader file: {}\n", filepath));
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

GLuint compile_shader(GLuint type, char const* source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);

  int success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char info_log[512];
    glGetShaderInfoLog(shader, 512, NULL, info_log);
    throw std::runtime_error(std::format("[compile_shader]: Shader compilation failed:\n{}\n", info_log));
  }
  return shader;
}

GLuint create_shader_program(char const* vert_path, char const* frag_path, char const* name) {
  std::string vert_source = load_shader_source(vert_path);
  GLuint vert_shader = compile_shader(GL_VERTEX_SHADER, vert_source.c_str());

  std::string frag_source = load_shader_source(frag_path);
  GLuint frag_shader = compile_shader(GL_FRAGMENT_SHADER, frag_source.c_str());

  GLuint program = glCreateProgram();
  glAttachShader(program, vert_shader);
  glAttachShader(program, frag_shader);
  glLinkProgram(program);

  int success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char info_log[512];
    glGetProgramInfoLog(program, 512, NULL, info_log);
    throw std::runtime_error(
        std::format("[create_shader_program]: {} shader program linking failed:\n{}\n", name, info_log));
  }

  glDeleteShader(vert_shader);
  glDeleteShader(frag_shader);

  return program;
}

} // namespace

Renderer::Renderer() {
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  _contourShaderProgram = create_shader_program(CONTOUR_VERT_SHADER_PATH, CONTOUR_FRAG_SHADER_PATH, "Contour");
  _mapShaderProgram = create_shader_program(MAP_VERT_SHADER_PATH, MAP_FRAG_SHADER_PATH, "Map");
  _circleShaderProgram = create_shader_program(CIRCLE_VERT_SHADER_PATH, CIRCLE_FRAG_SHADER_PATH, "Circle");

  initializeShaders();
}

void Renderer::initializeShaders() {
  glUseProgram(_mapShaderProgram);
  GLint min_value_loc = glGetUniformLocation(_mapShaderProgram, "minValue");
  glUniform1f(min_value_loc, HeightGrid::HEIGHT_RANGE.min);
  GLint max_value_loc = glGetUniformLocation(_mapShaderProgram, "maxValue");
  glUniform1f(max_value_loc, HeightGrid::HEIGHT_RANGE.max);

  _contourTLoc = glGetUniformLocation(_contourShaderProgram, "t");
}

Renderer::~Renderer() {
  glDeleteProgram(_contourShaderProgram);
  glDeleteProgram(_mapShaderProgram);
  glDeleteProgram(_circleShaderProgram);
}

void Renderer::clear() { glClear(GL_COLOR_BUFFER_BIT); }

void Renderer::renderContours(ContourLayer const& contours) {
  glUseProgram(_contourShaderProgram);
  glLineWidth(CONTOUR_LINE_WIDTH);
  glBindVertexArray(contours.vao());

  GLint last_offset = 0;

  for (size_t i = 0; i < static_cast<size_t>(contours.count()); ++i) {
    glUniform1f(_contourTLoc, contours.gradients().at(i));
    // We need a x2 to get the actual vertex count
    glDrawArrays(GL_LINES, last_offset, 2 * contours.offsets().at(i) - last_offset);
    last_offset = 2 * contours.offsets().at(i);
  }
}

void Renderer::renderMap(MapLayer const& map) {
  glUseProgram(_mapShaderProgram);
  glBindVertexArray(map.vao());
  glDrawElements(GL_TRIANGLES, map.indexCount(), GL_UNSIGNED_INT, 0);
}

void Renderer::renderCircle(CircleLayer const& circle) {
  CircleLayer::Parameters parameters = circle.parameters();
  if (!parameters.visible) {
    return;
  }

  glUseProgram(_circleShaderProgram);

  GLint center_loc = glGetUniformLocation(_circleShaderProgram, "centerGrid");
  glUniform2f(center_loc, parameters.circle.center.x, parameters.circle.center.y);

  GLint radius_loc = glGetUniformLocation(_circleShaderProgram, "radiusGrid");
  glUniform1f(radius_loc, parameters.circle.radius);

  glLineWidth(CIRCLE_LINE_WIDTH);
  glBindVertexArray(circle.vao());
  glDrawArrays(GL_LINE_LOOP, 0, CircleLayer::CIRCLE_SEGMENTS);
}

void Renderer::setCropRegion(ApplicationState const& state) {
  GLuint programs[] = {_contourShaderProgram, _mapShaderProgram, _circleShaderProgram};
  for (GLuint program : programs) {
    glUseProgram(program);
    GLint crop_min_loc = glGetUniformLocation(program, "cropMin");
    glUniform2f(crop_min_loc, state.crop.min.x, state.crop.min.y);
    GLint crop_max_loc = glGetUniformLocation(program, "cropMax");
    glUniform2f(crop_max_loc, state.crop.max.x, state.crop.max.y);
  }
}

void Renderer::render(std::shared_ptr<Scene> scene, ApplicationState const& state) {
  clear();
  setCropRegion(state);

  renderMap(scene->map);
  renderContours(scene->contour);
  renderCircle(scene->circle);
}