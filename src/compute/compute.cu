#include "compute/compute.h"
#include "compute/kernels/kernels.h"

void compute_contour(std::unique_ptr<MappedGpuResources> resources, Size grid_size, float threshold) {
  marching_squares<<<1, 1>>>(resources->heights(), resources->contours(), grid_size, threshold);
  cudaDeviceSynchronize();
}

void modify_height(std::unique_ptr<MappedGpuResources> resources, Size grid_size, Range height_range,
                   BrushDab brush_dab, float contour_threshold) {

  smoothstep<<<1, 1>>>(resources->heights(), grid_size, height_range, brush_dab);
  cudaDeviceSynchronize();

  marching_squares<<<1, 1>>>(resources->heights(), resources->contours(), grid_size, contour_threshold);
  cudaDeviceSynchronize();
}
