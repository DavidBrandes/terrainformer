#include "common/device.h"
#include "compute/compute.h"
#include "compute/kernels/kernels.h"
#include "device.h"

#include <functional>
#include <iostream>
#include <stdexcept>

namespace perf {

MappedHeightGrid::MappedHeightGrid(HeightGrid height_grid) : HeightGrid(std::move(height_grid)) {
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&mappedHeights), size.height * size.width * sizeof(float));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to malloc for map heights");
  }

  err = cudaMemcpy(mappedHeights, heights.data(), size.height * size.width * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(mappedHeights);
    throw std::runtime_error("Failed to copy map heights to device");
  }
}

MappedHeightGrid::~MappedHeightGrid() {
  cudaError_t err = cudaFree(mappedHeights);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to free mapped heights" << std::endl;
  }
}

ScratchBuffer::ScratchBuffer(int size) {
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(float));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to malloc for scratch buffer");
  }
}

ScratchBuffer::~ScratchBuffer() {
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to free scratch buffer" << std::endl;
  }
}

float benchmark(std::function<void()> func, int iterations, int warmup_iterations) {
  for (int i = 0; i < warmup_iterations; ++i) {
    func();
  }

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int i = 0; i < iterations; ++i) {
    func();
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  return ms / iterations;
}

void launch_smoothstep(MappedHeightGrid const& height_grid, ScratchBuffer const& scratch_buffer, BrushDab brush_dab) {
  Region region = brush_dab_region(brush_dab, GRID_CONFIG.size);

  if (region.empty()) {
    return;
  }

  Buffers heights{.src = height_grid.mappedHeights, .dst = scratch_buffer.ptr};

  dim3 block_dim(32, 4);
  dim3 grid_dim((region.size.width + block_dim.x - 1) / block_dim.x,
                (region.size.height + block_dim.y * SMOOTHSTEP_ROW_COARSE - 1) / (block_dim.y * SMOOTHSTEP_ROW_COARSE));
  smoothstep<<<grid_dim, block_dim>>>(heights, height_grid.size, region.origin, height_grid.HEIGHT_RANGE, brush_dab);
  cudaDeviceSynchronize();
}

} // namespace perf
