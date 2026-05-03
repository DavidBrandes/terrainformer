#pragma once

#include "app/requests.h"
#include "app/state.h"
#include "compute/resources.h"
#include "controller/scene.h"
#include "graphics/renderer.h"
#include "graphics/scene.h"
#include "ui/gui.h"
#include "ui/window.h"
#include "utils/config.h"

class Application {
public:
  Application(Config const& config);

  Application(Application const&) = delete;
  Application& operator=(Application const&) = delete;

  Application(Application&&) = delete;
  Application& operator=(Application&&) = delete;

  void initialize(ApplicationState& state);
  void processInput(ApplicationState& state, ApplicationRequests& requests);
  void update(ApplicationState& state, ApplicationRequests& requests);
  void render(ApplicationState const& state);
  void run();

private:
  void handleRequests(ApplicationRequests& requests);

  Window _window;
  GUI _gui;
  Renderer _renderer;
  SceneController _sceneController;
  std::shared_ptr<Scene> _scene;
  std::shared_ptr<compute::GpuResources> _resources;
};