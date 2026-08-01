#pragma once

#include "graphics/scene.h"

#include <memory>

struct cudaGraphicsResource;

namespace compute {
class MappedGpuResources;

class GpuResources : public std::enable_shared_from_this<GpuResources> {
public:
  ~GpuResources();

  GpuResources(GpuResources const&) = delete;
  GpuResources& operator=(GpuResources const&) = delete;

  GpuResources(GpuResources&&) = delete;
  GpuResources& operator=(GpuResources&&) = delete;

  static std::shared_ptr<GpuResources> make(std::shared_ptr<Scene> scene);

  std::shared_ptr<compute::MappedGpuResources> map();
  std::shared_ptr<Scene> scene() const { return _scene; }

private:
  GpuResources(cudaGraphicsResource**, std::shared_ptr<Scene>);

  friend compute::MappedGpuResources;

  std::shared_ptr<Scene> _scene;
  cudaGraphicsResource* _resources[2]; // [0] = contour VBO, [1] = map heights VBO
  int* _contourSegmentCount;
  float* _thresholds;
  int _thresholdCount;
};

} // namespace compute