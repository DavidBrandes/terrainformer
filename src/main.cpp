#include "app/app.h"
#include "utils/config.h"

int main() {
  Config config = load_config();

  Application app{config};
  app.run();
}