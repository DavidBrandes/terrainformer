#include "compute/resources.h"

#include "compute/mapped_resources.cuh"
#include "compute/utils.cuh"
#include "types.cuh"

namespace compute {

MappedGpuResources::MappedGpuResources(std::shared_ptr<GpuResources> parent) : _parent(parent) {
  static_assert(sizeof(Segment) == 4 * sizeof(float));
  static_assert(alignof(Segment) == alignof(float));

  size_t num_bytes;

  CUDA_CHECK(cudaGraphicsMapResources(2, parent->_resources));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&_contourSegments, &num_bytes, _parent->_resources[0]));
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&_heights, &num_bytes, _parent->_resources[1]));
}

Segments MappedGpuResources::contours() const {
  return Segments{.segments = _contourSegments,
                  .count = _parent->_contourSegmentCount,
                  .maxCount = _parent->_scene->contour.maxSegments()};
}

ConstrainedGrid MappedGpuResources::heights() const {
  Grid grid{.values = _heights, .size = _parent->_scene->map.size()};
  ConstrainedGrid constrained_grid{.grid = grid, .range = _parent->_scene->map.range()};

  return constrained_grid;
}

MappedGpuResources::~MappedGpuResources() { CUDA_CHECK(cudaGraphicsUnmapResources(2, _parent->_resources)); }

} // namespace compute