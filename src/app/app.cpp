#include "app/app.h"

#include "app.h"
#include "controller/scene.h"
#include "utils/height_grid.h"

Application::Application(Config const& config)
    : _window(Window{config}), _gui(GUI{_window}), _renderer(Renderer{}), _sceneController(SceneController{config}) {

  HeightGrid height_grid = make_height_grid(config.grid);
  _scene = make_scene(config, height_grid);
  _resources = compute::GpuResources::make(_scene);
}

void Application::initialize(ApplicationState& state) { _sceneController.initialize(_resources, state); }

void Application::processInput(ApplicationState& state, ApplicationRequests& requests) {
  _window.processInput(state, requests);
}

void Application::update(ApplicationState& state, ApplicationRequests& requests) {
  _gui.prepare(requests);
  handleRequests(requests);
  _sceneController.update(_resources, state);
}

void Application::render(ApplicationState const& state) {
  _renderer.render(_scene, state);
  _gui.render();
  _window.present();
}

void Application::run() {
  ApplicationState state;
  ApplicationRequests requests;

  initialize(state);

  while (_window.isOpen()) {
    processInput(state, requests);
    update(state, requests);
    render(state);
  }
}

void Application::handleRequests(ApplicationRequests& requests) {
  if (requests.shutdown) {
    _window.close();
    requests.shutdown = false;
  }

  if (requests.toggleFullscreen) {
    _window.toggleFullscreen();
    requests.toggleFullscreen = false;
  }
}
