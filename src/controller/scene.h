#pragma once

#include "app/state.h"
#include "compute/resources.h"
#include "utils/config.h"
#include "utils/geometry.h"

class SceneController {
public:
  SceneController(Config const& config);

  void initialize(std::shared_ptr<GpuResources> resources, ApplicationState& state);
  void update(std::shared_ptr<GpuResources> resources, ApplicationState& state);

private:
  void updateCropState(ApplicationState& state);
  void updateVertices(VertexLayer& vertex, ApplicationState& state);

  Size _gridSize;
  Range _effectRadius;
  float _heightBrushIntensity;
  SceneConfig _initialSceneConfig;
};