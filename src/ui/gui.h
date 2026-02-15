#pragma once

#include "ui/window.h"
#include "app/requests.h"
#include "app/state.h"

class GUI {
public:
  GUI(Window& window);
  ~GUI();

  GUI(GUI const&) = delete;
  GUI& operator=(GUI const&) = delete;

  GUI(GUI&&) = delete;
  GUI& operator=(GUI&&) = delete;

  void prepare(ApplicationState& state, ApplicationRequests& requests);
  void render();

private:
  void createButtonRow(ApplicationState& state);
  void createCloseButton(ApplicationRequests& requests);
};
