#include "app/app.h"
#include "app/requests.h"
#include "app/state.h"
#include "utils/config.h"

#include <benchmark/benchmark.h>

// TODO: simulate app run
static void BM_FrameLoop(benchmark::State& benchmark_state) {
  Config config = load_config();
  config.window.headless = true;

  Application app(config);

  ApplicationState app_state;
  ApplicationRequests requests;

  app.initialize(app_state);

  for (auto _ : benchmark_state) {
    app.processInput(app_state, requests);
    app.update(app_state, requests);
    app.render(app_state);
  }
}

// TODO: fix repititions, iterations, output format
BENCHMARK(BM_FrameLoop)->Iterations(100)->Repetitions(5);
