#include "common/device.h"
#include "common/setup.h"

#include <CLI/CLI.hpp>
#include <format>
#include <iostream>

namespace {

constexpr int BENCHMARK_BRUSH_DAB_STEPS = 3;

float ms_to_us(float ms) { return ms * 1000; }

void benchmark_smoothstep(int iterations, int warmup_iterations) {
  std::cout << std::format("Benchmarking smoothstep kernel with grid dimensions {} x {}", perf::GRID_CONFIG.size.height,
                           perf::GRID_CONFIG.size.width)
            << std::endl;

  float total_ms = 0;

  perf::MappedHeightGrid height_grid = perf::make_height_grid();
  perf::ScratchBuffer scratch_buffer{height_grid.size.height * height_grid.size.width};

  for (auto const& brush_dab : perf::generate_brush_dabs(BENCHMARK_BRUSH_DAB_STEPS)) {
    float duration_ms = perf::benchmark([&]() { perf::launch_smoothstep(height_grid, scratch_buffer, brush_dab); },
                                        iterations, warmup_iterations);
    total_ms += duration_ms;

    std::cout << std::format("  pos ({:.1f}, {:.1f})  radius {:.1f}  ->  {:.1f} µs", brush_dab.circle.center.x,
                             brush_dab.circle.center.y, brush_dab.circle.radius, ms_to_us(duration_ms))
              << std::endl;
  }

  std::cout << std::format("Total: {:.1f} µs", ms_to_us(total_ms)) << std::endl;
}

void add_smoothstep_subcommand(CLI::App& app, int& iterations, int& warmup_iterations) {
  auto* cmd = app.add_subcommand("smoothstep", "Benchmark the smoothstep kernel");
  cmd->callback([&] { benchmark_smoothstep(iterations, warmup_iterations); });
}

} // namespace

int main(int argc, char** argv) {
  CLI::App app{"terrainformer benchmark"};
  app.require_subcommand(1);

  int iterations = 1000;
  int warmup_iterations = 10;
  app.add_option("--iterations", iterations, "Number of benchmark iterations")->capture_default_str();
  app.add_option("--warmup-iterations", warmup_iterations, "Number of warmup iterations")->capture_default_str();

  add_smoothstep_subcommand(app, iterations, warmup_iterations);

  CLI11_PARSE(app, argc, argv);

  return 0;
}