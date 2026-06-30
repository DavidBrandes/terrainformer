#pragma once

#include "app/requests.h"
#include "app/state.h"
#include "utils/config.h"
#include "utils/grid.h"

// Forward declaration to not include glfw3.h
struct GLFWwindow;

class Window {
public:
  ~Window();
  Window(Config const& config);

  Window(Window const&) = delete;
  Window& operator=(Window const&) = delete;

  Window(Window&&) = delete;
  Window& operator=(Window&&) = delete;

  bool isOpen() const;
  void processInput(ApplicationState& state, ApplicationRequests& requests);
  void present();
  void close();
  void toggleFullscreen();

  GLFWwindow* getWindow() const;

private:
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

  GLFWwindow* _window;
  bool _isFullscreen;
  Size _defaultSize;
  float _scrollInterval;
  float _scroll;
  int _windowedXPos = 0;
  int _windowedYPos = 0;

  static constexpr char const* WINDOW_TITLE = "Terrain Former";
};