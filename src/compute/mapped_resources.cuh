#pragma once

#include <memory>

#include "compute/types.cuh"

namespace compute {

class GpuResources;

class MappedGpuResources {
public:
  ~MappedGpuResources();

  MappedGpuResources(MappedGpuResources const&) = delete;
  MappedGpuResources& operator=(MappedGpuResources const&) = delete;

  MappedGpuResources(MappedGpuResources&&) = delete;
  MappedGpuResources& operator=(MappedGpuResources&&) = delete;

  Grid heights() const;
  Segments contours() const;

private:
  friend GpuResources;

  MappedGpuResources(std::shared_ptr<GpuResources> parent);
  std::shared_ptr<GpuResources> _parent;

  float* _heights;
  Segment* _contourSegments;
};

} // namespace compute