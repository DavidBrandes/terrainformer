#include "controller/scene.h"

#include "app/state.h"
#include "compute/compute.h"
#include "utils/config.h"
#include "utils/height_grid.h"
#include "utils/types.h"

#include <algorithm>
#include <vector>

namespace {
std::vector<float> compute_thresholds(Config const& config) {
  std::vector<float> thresholds;

  for (int i = 0; i < config.scene.contourCount; ++i) {
    // We do not want any thresholds at the bounds
    float t = static_cast<float>(i + 1) / static_cast<float>(config.scene.contourCount + 1);
    float threshold = HeightGrid::HEIGHT_RANGE.min + t * HeightGrid::HEIGHT_RANGE.span();
    thresholds.push_back(threshold);
  }

  return thresholds;
}

bool is_active(ApplicationState const& state) {
  if (state.tool.activeTool == ToolState::Tool::NONE) {
    return false;
  }

  if (state.mouse.pixel.x < 0 || static_cast<float>(state.window.size.width) < state.mouse.pixel.x) {
    return false;
  }

  if (state.mouse.pixel.y < 0 || static_cast<float>(state.window.size.height) < state.mouse.pixel.y) {
    return false;
  }

  return true;
}

void update_circle_layer(std::shared_ptr<compute::GpuResources>& resources, Circle circle, bool visible) {
  resources->scene()->circle.update(CircleLayer::Parameters{.circle = circle, .visible = visible});
}
} // namespace

SceneController::SceneController(Config const& config)
    : _gridSize(config.grid.size), _effectRadius(config.tool.effectRadius),
      _heightBrushIntensity(HeightGrid::HEIGHT_RANGE.span() * config.tool.heightBrushSensitivity),
      _initialSceneConfig(config.scene) {
  float min_grid_dim =
      std::min({static_cast<float>(_gridSize.width), static_cast<float>(_gridSize.height), _effectRadius.max});
  float max_radius = static_cast<float>(min_grid_dim - 1) / 2;
  _radiusRange = std::min(max_radius, _effectRadius.max) - _effectRadius.min;
  _thresholds = compute_thresholds(config);
}

void SceneController::updateCropState(ApplicationState& state) const {
  state.crop.min = Point{.x = 0, .y = 0};
  state.crop.max = Point{.x = static_cast<float>(_gridSize.width - 1), .y = static_cast<float>(_gridSize.height - 1)};

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

Circle SceneController::computeEffectCircle(ApplicationState const& state) const {
  float x =
      (state.mouse.pixel.x / static_cast<float>(state.window.size.width)) * static_cast<float>(_gridSize.width - 1);
  float y =
      (state.mouse.pixel.y / static_cast<float>(state.window.size.height)) * static_cast<float>(_gridSize.height - 1);

  float scroll_normalized = (state.mouse.scroll - MouseState::SCROLL.min) / MouseState::SCROLL.span();
  float radius = _effectRadius.min + scroll_normalized * _radiusRange;

  return Circle{.center = Point{.x = x, .y = y}, .radius = radius};
}

void SceneController::shift(std::shared_ptr<compute::GpuResources>& resources, ApplicationState const& state,
                            Circle effect_circle) const {
  float intensity = state.mouse.button == MouseState::Button::LEFT ? _heightBrushIntensity : -_heightBrushIntensity;
  BrushDab brush_dab{.circle = effect_circle, .intensity = intensity};

  compute::Result result = modify_height(resources->map(), HeightGrid::HEIGHT_RANGE, brush_dab, _thresholds);

  if (result.modified) {
    resources->scene()->contour.update(std::move(result.contourOffsets));
  }
}

void SceneController::update(std::shared_ptr<compute::GpuResources> resources, ApplicationState& state) {
  updateCropState(state);

  if (!is_active(state)) {
    update_circle_layer(resources, Circle{}, false);
    return;
  }

  Circle effect_circle = computeEffectCircle(state);
  update_circle_layer(resources, effect_circle, true);

  if (state.mouse.button == MouseState::Button::NONE) {
    return;
  }

  if (state.tool.activeTool == ToolState::Tool::SHIFT) {
    shift(resources, state, effect_circle);
    return;
  }
}

void SceneController::initialize(std::shared_ptr<compute::GpuResources> resources, ApplicationState& state) {
  state.scene.preserveAspectRatio = _initialSceneConfig.preserveAspectRatio;

  compute::Result result = compute_contour(resources->map(), _thresholds);

  if (result.modified) {
    resources->scene()->contour.update(std::move(result.contourOffsets));
  }
}
