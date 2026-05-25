#pragma once

#include "app/requests.h"
#include "ui/window.h"

class GUI {
public:
  GUI(Window& window);
  ~GUI();

  GUI(GUI const&) = delete;
  GUI& operator=(GUI const&) = delete;

  GUI(GUI&&) = delete;
  GUI& operator=(GUI&&) = delete;

  void prepare(ApplicationRequests& requests);
  void render();

private:
  void createCloseButton(ApplicationRequests& requests);
};
