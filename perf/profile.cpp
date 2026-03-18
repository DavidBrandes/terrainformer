
#include "common/device.h"
#include "common/setup.h"

#include <CLI/CLI.hpp>

namespace {
void add_smoothstep_subcommand(CLI::App& app) {
  auto* cmd = app.add_subcommand("smoothstep", "Smoothstep kernel");

  float x_normalized = 0.5f;
  float y_normalized = 0.5f;
  float radius_normalized = 0.5f;

  cmd->add_option("--x", x_normalized, "Normalized brush dab x")->capture_default_str();
  cmd->add_option("--y", y_normalized, "Normalized brush dab y")->capture_default_str();
  cmd->add_option("--radius", radius_normalized, "Normalized brush dab radius")->capture_default_str();

  cmd->callback([=] {
    Point center = perf::point_from_normalized(x_normalized, y_normalized);
    float radius = perf::radius_from_normalized(radius_normalized);
    BrushDab brush_dab = perf::make_brush_dab(center, radius);
    perf::MappedHeightGrid height_grid = perf::make_height_grid();
    perf::ScratchBuffer scratch_buffer{height_grid.size.height * height_grid.size.width};

    perf::launch_smoothstep(height_grid, scratch_buffer, brush_dab);
  });
}
} // namespace

int main(int argc, char** argv) {
  CLI::App app{"terrainformer profiler"};
  app.require_subcommand(1);

  add_smoothstep_subcommand(app);

  CLI11_PARSE(app, argc, argv);

  return 0;
}