#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "compute/types.cuh"

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = (call);                                                                                          \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                      \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

namespace compute {
int ceil_div(int n, int d);

Region aligned_brush_dab_region(BrushDab brush_dab, Size size);

} // namespace compute