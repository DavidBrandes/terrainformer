#include "compute/resources.h"

#include "compute/mapped_resources.cuh"
#include "compute/utils.cuh"
#include "types.cuh"

namespace compute {

MappedGpuResources::MappedGpuResources(std::shared_ptr<GpuResources> parent) : _parent(parent) {
  size_t num_bytes;

  CUDA_CHECK(cudaGraphicsMapResources(2, parent->_resources));
  // We here map to a float4 pointer, which will work as we define the GL resource containing a multiple of 4 float
  // elements
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&_contourSegments, &num_bytes, _parent->_resources[0]));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&_heights, &num_bytes, _parent->_resources[1]));
}

Segments MappedGpuResources::contours() const {
  return Segments{.values = _contourSegments,
                  .count = _parent->_contourSegmentCount,
                  .maxCount = _parent->_scene->contour.maxCount()};
}

ConstrainedGrid MappedGpuResources::constrainedHeights() const {
  Grid grid{.values = _heights, .size = _parent->_scene->map.size()};
  ConstrainedGrid constrained_grid{.grid = grid, .range = _parent->_scene->map.range()};

  return constrained_grid;
}

CGrid MappedGpuResources::heights() const { return CGrid{.values = _heights, .size = _parent->_scene->map.size()}; }

Thresholds MappedGpuResources::thresholds() const {
  return Thresholds{.values = _parent->_thresholds, .count = _parent->_thresholdCount};
}

MappedGpuResources::~MappedGpuResources() { CUDA_CHECK(cudaGraphicsUnmapResources(2, _parent->_resources)); }

} // namespace compute