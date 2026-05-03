#include "compute/resources.h"

#include <cuda_gl_interop.h>
#include <glad/gl.h>

#include "compute/mapped_resources.cuh"
#include "compute/utils.cuh"

namespace compute {

GpuResources::GpuResources(cudaGraphicsResource** resources, std::shared_ptr<Scene> scene) : _scene(scene) {
  _resources[0] = resources[0];
  _resources[1] = resources[1];

  CUDA_CHECK(cudaMalloc(&_contourSegmentCount, sizeof(int)));
}

std::shared_ptr<GpuResources> GpuResources::make(std::shared_ptr<Scene> scene) {
  cudaGraphicsResource* resources[2];

  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resources[0], scene->contour.vbo(), cudaGraphicsMapFlagsWriteDiscard));
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resources[1], scene->map.heightsVbo(), cudaGraphicsMapFlagsWriteDiscard));

  return std::shared_ptr<GpuResources>(new GpuResources{resources, scene});
}

GpuResources::~GpuResources() {
  CUDA_CHECK(cudaGraphicsUnregisterResource(_resources[0]));
  CUDA_CHECK(cudaGraphicsUnregisterResource(_resources[1]));

  CUDA_CHECK(cudaFree(_contourSegmentCount));
}

std::shared_ptr<compute::MappedGpuResources> GpuResources::map() {
  return std::shared_ptr<compute::MappedGpuResources>(new compute::MappedGpuResources{shared_from_this()});
}

} // namespace compute