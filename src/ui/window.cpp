#include "ui/window.h"

#include "app/state.h"

#include <glad/gl.h>
// ensure glad.h is included before glfw3.h
#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>

namespace {
void frame_buffersize_callback(GLFWwindow*, int width, int height) { glViewport(0, 0, width, height); }

GLFWmonitor* get_current_monitor(GLFWwindow* window) {
  int window_x, window_y, window_width, window_height;
  glfwGetWindowPos(window, &window_x, &window_y);
  glfwGetWindowSize(window, &window_width, &window_height);

  int monitor_count;
  GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);

  GLFWmonitor* target_monitor = glfwGetPrimaryMonitor();
  int max_overlap = 0;

  for (int i = 0; i < monitor_count; i++) {
    int monitor_x, monitor_y;
    glfwGetMonitorPos(monitors[i], &monitor_x, &monitor_y);

    GLFWvidmode const* mode = glfwGetVideoMode(monitors[i]);

    int overlap_x1 = std::max(window_x, monitor_x);
    int overlap_y1 = std::max(window_y, monitor_y);
    int overlap_x2 = std::min(window_x + window_width, monitor_x + mode->width);
    int overlap_y2 = std::min(window_y + window_height, monitor_y + mode->height);

    int overlap = std::max(0, overlap_x2 - overlap_x1) * std::max(0, overlap_y2 - overlap_y1);

    if (overlap > max_overlap) {
      max_overlap = overlap;
      target_monitor = monitors[i];
    }
  }

  return target_monitor;
}

void handle_keyboard_input(GLFWwindow* window, ApplicationState& state, ApplicationRequests& requests) {

  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
    requests.shutdown = true;
  }

  // We need to debounce the press to only detect it once
  static bool s_key_was_pressed = false;
  bool s_key_is_pressed = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
  if (s_key_is_pressed && !s_key_was_pressed) {
    state.tool.activeTool =
        (state.tool.activeTool == ToolState::Tool::SHIFT) ? ToolState::Tool::NONE : ToolState::Tool::SHIFT;
  }
  s_key_was_pressed = s_key_is_pressed;

  static bool f_key_was_pressed = false;
  bool f_key_is_pressed = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
  if (f_key_is_pressed && !f_key_was_pressed) {
    requests.toggleFullscreen = true;
  }
  f_key_was_pressed = f_key_is_pressed;

  static bool c_key_was_pressed = false;
  bool c_key_is_pressed = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
  if (c_key_is_pressed && !c_key_was_pressed) {
    state.scene.preserveAspectRatio = !state.scene.preserveAspectRatio;
  }
  c_key_was_pressed = c_key_is_pressed;
}

void handle_mouse_input(GLFWwindow* window, ApplicationState& state) {
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  state.mouse.pixel.x = static_cast<float>(xpos);
  state.mouse.pixel.y = static_cast<float>(ypos);

  int left_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  int right_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

  if (left_state == GLFW_PRESS && state.mouse.button == MouseState::Button::LEFT) {
    // prioritize current press
  } else if (right_state == GLFW_PRESS && state.mouse.button == MouseState::Button::RIGHT) {
    // prioritize current press
  } else if (left_state == GLFW_PRESS) {
    state.mouse.button = MouseState::Button::LEFT;
  } else if (right_state == GLFW_PRESS) {
    state.mouse.button = MouseState::Button::RIGHT;
  } else {
    state.mouse.button = MouseState::Button::NONE;
  }
}
} // namespace

void Window::scrollCallback(GLFWwindow* window, double, double yoffset) {

  Window* app = static_cast<Window*>(glfwGetWindowUserPointer(window));
  if (app) {
    if (yoffset == 0) {
      return;
    } else if (yoffset > 0) {
      app->_scroll += app->_scrollInterval;
    } else {
      app->_scroll -= app->_scrollInterval;
    }

    app->_scroll = std::clamp(app->_scroll, MouseState::SCROLL.min, MouseState::SCROLL.max);
  }
}

Window::Window(Config const& config) : _isFullscreen(config.window.fullscreen), _defaultSize(config.window.size) {
  _scrollInterval = MouseState::SCROLL.span() / static_cast<float>(config.tool.scrollSteps - 1);
  _scroll = MouseState::SCROLL.min + static_cast<float>(config.tool.scrollSteps / 2) * _scrollInterval;

  if (!glfwInit()) {
    throw std::runtime_error("[Window::Window]: Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, config.window.headless ? GLFW_FALSE : GLFW_TRUE);

  GLFWmonitor* monitor = _isFullscreen ? glfwGetPrimaryMonitor() : nullptr;
  int width = config.window.size.width;
  int height = config.window.size.height;
  if (_isFullscreen) {
    GLFWvidmode const* mode = glfwGetVideoMode(monitor);
    width = mode->width;
    height = mode->height;
  }
  _window = glfwCreateWindow(width, height, WINDOW_TITLE, monitor, nullptr);
  if (!_window) {
    glfwTerminate();
    throw std::runtime_error("[Window::Window]: Failed to create GLFW window");
  }

  glfwMakeContextCurrent(_window);
  glfwSetFramebufferSizeCallback(_window, frame_buffersize_callback);
  glfwSetWindowUserPointer(_window, this);
  glfwSetScrollCallback(_window, scrollCallback);

  if (!gladLoadGL(glfwGetProcAddress)) {
    glfwTerminate();
    throw std::runtime_error("[Window::Window]: Failed to initialize GLAD");
  }
}

Window::~Window() {
  if (_window) {
    glfwDestroyWindow(_window);
  }

  glfwTerminate();
}

bool Window::isOpen() const { return !glfwWindowShouldClose(_window); }

void Window::processInput(ApplicationState& state, ApplicationRequests& requests) {
  glfwPollEvents();

  state.mouse.scroll = _scroll;

  glfwGetWindowSize(_window, &state.window.size.width, &state.window.size.height);

  handle_keyboard_input(_window, state, requests);
  handle_mouse_input(_window, state);
}

void Window::present() { glfwSwapBuffers(_window); }
void Window::close() { glfwSetWindowShouldClose(_window, true); }

GLFWwindow* Window::getWindow() const { return _window; }

void Window::toggleFullscreen() {
  if (_isFullscreen) {
    glfwSetWindowMonitor(_window, nullptr, _windowedXPos, _windowedYPos, _defaultSize.width, _defaultSize.height, 0);
  } else {
    glfwGetWindowPos(_window, &_windowedXPos, &_windowedYPos);

    GLFWmonitor* monitor = get_current_monitor(_window);
    GLFWvidmode const* mode = glfwGetVideoMode(monitor);
    glfwSetWindowMonitor(_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
  }

  _isFullscreen = !_isFullscreen;
}