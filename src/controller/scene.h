#pragma once

#include "app/state.h"
#include "compute/resources.h"
#include "utils/config.h"
#include "utils/types.h"

#include <vector>

class SceneController {
public:
  SceneController(Config const& config);

  void initialize(std::shared_ptr<compute::GpuResources> resources, ApplicationState& state);
  void update(std::shared_ptr<compute::GpuResources> resources, ApplicationState& state);

private:
  void updateCropState(ApplicationState& state) const;
  Circle computeEffectCircle(ApplicationState const& state) const;
  void shift(std::shared_ptr<compute::GpuResources>& resources, ApplicationState const& state,
             Circle effect_circle) const;

  Size _gridSize;
  Range _effectRadius;
  float _radiusRange;
  float _heightBrushIntensity;
  SceneConfig _initialSceneConfig;
  std::vector<float> _thresholds; // TODO define in const memory
};