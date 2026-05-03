#include "controller/scene.h"

#include "app/state.h"
#include "compute/compute.h"
#include "utils/config.h"
#include "utils/height_grid.h"
#include "utils/types.h"

#include <algorithm>

constexpr float CONTOUR_THRESHOLD = 0.0f; // TODO define somehow else

SceneController::SceneController(Config const& config)
    : _gridSize(config.grid.size), _effectRadius(config.tool.effectRadius),
      _heightBrushIntensity(HeightGrid::HEIGHT_RANGE.span() * config.tool.heightBrushSensitivity),
      _initialSceneConfig(config.scene) {
  float min_grid_dim =
      std::min({static_cast<float>(_gridSize.width), static_cast<float>(_gridSize.height), _effectRadius.max});
  float max_radius = static_cast<float>(min_grid_dim - 1) / 2;
  _radiusRange = std::min(max_radius, _effectRadius.max) - _effectRadius.min;
}

void SceneController::updateCropState(ApplicationState& state) {
  state.crop.min = Point{0, 0};
  state.crop.max = Point{_gridSize.width - 1, _gridSize.height - 1};

  if (!state.scene.preserveAspectRatio) {
    return;
  }

  float window_aspect_ratio =
      static_cast<float>(state.window.size.height) / static_cast<float>(state.window.size.width);
  float grid_aspect_ratio = static_cast<float>(_gridSize.height - 1) / static_cast<float>(_gridSize.width - 1);

  // We need to crop along the height/ y-axis
  if (window_aspect_ratio < grid_aspect_ratio) {
    float target_height = static_cast<float>(_gridSize.width - 1) * window_aspect_ratio;
    float offset = (static_cast<float>(_gridSize.height - 1) - target_height) / 2;

    state.crop.min.y += offset;
    state.crop.max.y -= offset;

    return;
  }

  // We need to crop along the width / x-axis
  if (window_aspect_ratio > grid_aspect_ratio) {
    float target_width = static_cast<float>(_gridSize.height - 1) / window_aspect_ratio;
    float offset = (static_cast<float>(_gridSize.width - 1) - target_width) / 2;

    state.crop.min.x += offset;
    state.crop.max.x -= offset;

    return;
  }
}

void SceneController::updateVertices(VertexLayer& vertex, ApplicationState& state) {
  vertex.update(state.scene.showVertices);
}

// TODO split into smaller chunks
void SceneController::update(std::shared_ptr<compute::GpuResources> resources, ApplicationState& state) {
  updateCropState(state);
  updateVertices(resources->scene()->vertex, state);

  if (state.tool.activeTool == ToolState::Tool::NONE) {
    resources->scene()->circle.update(CircleLayer::Parameters{.circle = Circle{}, .visible = false});
    return;
  }

  if (state.mouse.pixel.x < 0 || static_cast<float>(state.window.size.width) < state.mouse.pixel.x) {
    resources->scene()->circle.update(CircleLayer::Parameters{.circle = Circle{}, .visible = false});
    return;
  }

  if (state.mouse.pixel.y < 0 || static_cast<float>(state.window.size.height) < state.mouse.pixel.y) {
    resources->scene()->circle.update(CircleLayer::Parameters{.circle = Circle{}, .visible = false});
    return;
  }

  float x =
      (state.mouse.pixel.x / static_cast<float>(state.window.size.width)) * static_cast<float>(_gridSize.width - 1);
  float y =
      (state.mouse.pixel.y / static_cast<float>(state.window.size.height)) * static_cast<float>(_gridSize.height - 1);

  float scroll_normalized = (state.mouse.scroll - MouseState::SCROLL.min) / MouseState::SCROLL.span();
  float radius = _effectRadius.min + scroll_normalized * _radiusRange;

  resources->scene()->circle.update(CircleLayer::Parameters{
      .circle = {Point{x, y}, radius},
      .visible = true,
  });

  if (state.mouse.button == MouseState::Button::NONE) {
    return;
  }

  if (state.tool.activeTool == ToolState::Tool::A) {
    float intensity = state.mouse.button == MouseState::Button::LEFT ? _heightBrushIntensity : -_heightBrushIntensity;
    BrushDab brush_dab{.circle = {Point{x, y}, radius}, .intensity = intensity};

    compute::Result result = modify_height(resources->map(), HeightGrid::HEIGHT_RANGE, brush_dab, CONTOUR_THRESHOLD);

    resources->scene()->contour.update(result.contourSegmentCount);
  }
}

void SceneController::initialize(std::shared_ptr<compute::GpuResources> resources, ApplicationState& state) {
  state.scene.preserveAspectRatio = _initialSceneConfig.preserveAspectRatio;
  state.scene.showVertices = _initialSceneConfig.showVertices;

  compute::Result result = compute_contour(resources->map(), CONTOUR_THRESHOLD);

  resources->scene()->contour.update(result.contourSegmentCount);
}
