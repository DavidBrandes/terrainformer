#include "compute/resources.h"
#include "utils/height_grid.h"

#include <compute/parameters.h>
#include <cuda_gl_interop.h>
#include <glad/gl.h>

#include <iostream>

GpuResources::GpuResources(cudaGraphicsResource** resources, std::shared_ptr<Scene> scene) : _scene(scene) {
  _resources[0] = resources[0];
  _resources[1] = resources[1];
}

std::shared_ptr<GpuResources> GpuResources::make(std::shared_ptr<Scene> scene) {
  cudaGraphicsResource* resources[2];

  cudaError_t err = cudaGraphicsGLRegisterBuffer(&resources[0], scene->contour.vbo(), cudaGraphicsMapFlagsWriteDiscard);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to register OpenGL buffer for Contour");
  }

  err = cudaGraphicsGLRegisterBuffer(&resources[1], scene->map.heightsVbo(), cudaGraphicsMapFlagsWriteDiscard);
  if (err != cudaSuccess) {
    cudaGraphicsUnregisterResource(resources[0]);
    throw std::runtime_error("Failed to register OpenGL buffer for Map heights");
  }

  return std::shared_ptr<GpuResources>(new GpuResources{resources, scene});
}

GpuResources::~GpuResources() {
  cudaError_t err = cudaGraphicsUnregisterResource(_resources[0]);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to unregister contour graphics resource" << std::endl;
  }
  err = cudaGraphicsUnregisterResource(_resources[1]);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to unregister map graphics resource" << std::endl;
  }
}

std::unique_ptr<MappedGpuResources> GpuResources::map() {
  cudaError_t err = cudaGraphicsMapResources(2, _resources);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to map graphic resources");
  }

  return std::unique_ptr<MappedGpuResources>(new MappedGpuResources{shared_from_this()});
}

MappedGpuResources::MappedGpuResources(std::shared_ptr<GpuResources> parent) : _parent(parent) {
  size_t num_bytes;

  cudaError_t err = cudaGraphicsResourceGetMappedPointer((void**)&_contours, &num_bytes, _parent->_resources[0]);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get contour segments mapped pointer");
  }

  err = cudaGraphicsResourceGetMappedPointer((void**)&_heights, &num_bytes, _parent->_resources[1]);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get heights mapped pointer");
  }
}

MappedGpuResources::~MappedGpuResources() {
  cudaError_t err = cudaGraphicsUnmapResources(2, _parent->_resources);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to unmap graphic resources" << std::endl;
  }
}

