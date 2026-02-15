#pragma once

#include "compute/parameters.h"
#include "graphics/scene.h"

#include <memory>

struct cudaGraphicsResource;
class MappedGpuResources;

class GpuResources : public std::enable_shared_from_this<GpuResources> {
public:
  ~GpuResources();

  GpuResources(GpuResources const&) = delete;
  GpuResources& operator=(GpuResources const&) = delete;

  GpuResources(GpuResources&&) = delete;
  GpuResources& operator=(GpuResources&&) = delete;

  static std::shared_ptr<GpuResources> make(std::shared_ptr<Scene> scene);

  std::unique_ptr<MappedGpuResources> map();
  std::shared_ptr<Scene> scene() const { return _scene; }

private:
  GpuResources(cudaGraphicsResource**, std::shared_ptr<Scene>);

  friend MappedGpuResources;

  std::shared_ptr<Scene> _scene;
  cudaGraphicsResource* _resources[2]; // [0] = contour VBO, [1] = map heights VBO
};

class MappedGpuResources {
public:
  ~MappedGpuResources();

  MappedGpuResources(MappedGpuResources const&) = delete;
  MappedGpuResources& operator=(MappedGpuResources const&) = delete;

  MappedGpuResources(MappedGpuResources&&) = delete;
  MappedGpuResources& operator=(MappedGpuResources&&) = delete;

  // TODO check if inlining made this better, compare with -flto flag
  float* heights() const { return _heights; }
  float* contours() const { return _contours; }

private:
  friend GpuResources;

  MappedGpuResources(std::shared_ptr<GpuResources> parent);
  std::shared_ptr<GpuResources> _parent;

  float* _heights;
  float* _contours;
};