#pragma once

#include "utils/geometry.h"
#include "utils/grid.h"
#include "utils/height_grid.h"

#include <driver_types.h>

#include <iomanip>
#include <iostream>
#include <string_view>

#include "compute/types.cuh"

namespace perf {

template <typename T>
struct GpuBuffer {
  GpuBuffer(int size) {
    this->size = size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
  }
  ~GpuBuffer() { CUDA_CHECK(cudaFree(data)); }

  GpuBuffer(GpuBuffer const& other) = delete;
  GpuBuffer& operator=(GpuBuffer const& other) = delete;

  GpuBuffer(GpuBuffer&& other) noexcept {
    data = other.data;
    size = other.size;
    other.data = nullptr;
  }

  GpuBuffer& operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
      CUDA_CHECK(cudaFree(data));

      data = other.data;
      size = other.size;
      other.data = nullptr;
    }

    return *this;
  }

  static GpuBuffer<T> fromVector(std::vector<T> const& vec) {
    GpuBuffer<T> buffer{static_cast<int>(vec.size())};
    CUDA_CHECK(cudaMemcpy(buffer.data, vec.data(), buffer.bytes(), cudaMemcpyHostToDevice));

    return buffer;
  }

  std::vector<T> toVector() const {
    std::vector<T> vec(size);
    CUDA_CHECK(cudaMemcpy(vec.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));

    return vec;
  }

  int bytes() const { return size * sizeof(T); }

  T* data;
  int size;
};

GpuBuffer<float> make_gpu_buffer(HeightGrid const& height_grid);

template <typename T = decltype([]() {}), typename U = decltype([]() {})>
void benchmark(T&& func, U&& setup = []() {}, cudaStream_t stream = 0) {
  constexpr int warmup_iterations = 5;
  constexpr int benchmark_iterations = 100;

  for (int i = 0; i < warmup_iterations; ++i) {
    setup();
    func();
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float total_ms = 0;

  for (int i = 0; i < benchmark_iterations; ++i) {
    setup();

    CUDA_CHECK(cudaEventRecord(start, stream));
    func();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  float avg_us = (total_ms / benchmark_iterations) * 1000;

  std::cout << "Average runtime across " << benchmark_iterations << " iterations: " << std::fixed
            << std::setprecision(2) << avg_us << " µs" << std::endl;
}

Point point_from_normalized(Size size, float x_normalized = 0.5f, float y_normalized = 0.5f);
float radius_from_normalized(Size size, float radius_normalized = 0.5f);
BrushDab make_brush_dab(Point center, float radius, float intensity = 0.01f);

void plot(std::vector<float> const& heights, Size size, std::vector<float4> const& segments = {},
          std::string_view title = "plot");

} // namespace perf