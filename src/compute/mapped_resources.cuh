#pragma once

#include <memory>

#include "compute/types.cuh"
#include "types.cuh"

namespace compute {

class GpuResources;

class MappedGpuResources {
public:
  ~MappedGpuResources();

  MappedGpuResources(MappedGpuResources const&) = delete;
  MappedGpuResources& operator=(MappedGpuResources const&) = delete;

  MappedGpuResources(MappedGpuResources&&) = delete;
  MappedGpuResources& operator=(MappedGpuResources&&) = delete;

  ConstrainedGrid constrainedHeights() const;
  CGrid heights() const;
  Segments contours() const;
  Thresholds thresholds() const;

private:
  friend GpuResources;

  MappedGpuResources(std::shared_ptr<GpuResources> parent);
  std::shared_ptr<GpuResources> _parent;

  float* _heights;
  float4* _contourSegments;
};

} // namespace compute