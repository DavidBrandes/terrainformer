#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "compute/utils.cuh"
#include "setup.cuh"

namespace perf {
GpuBuffer<float> make_gpu_buffer(HeightGrid const& height_grid) {
  GpuBuffer<float> buffer{height_grid.size.width * height_grid.size.height};
  CUDA_CHECK(cudaMemcpy(buffer.data, height_grid.heights.data(), buffer.size * sizeof(float), cudaMemcpyHostToDevice));

  return buffer;
}

Point point_from_normalized(Size size, float x_normalized, float y_normalized) {
  float x = x_normalized * (size.width - 1);
  float y = y_normalized * (size.height - 1);

  return Point{.x = x, .y = y};
}

float radius_from_normalized(Size size, float radius_normalized) {
  return (std::min(size.height, size.width) - 1) * radius_normalized;
}

BrushDab make_brush_dab(Point center, float radius, float intensity) {
  Circle circle{.center = center, .radius = radius};

  return BrushDab{.circle = circle, .intensity = intensity};
}

void plot(std::vector<float> const& heights, Size size, std::vector<compute::Segment> const& segments) {
  cv::Mat gray(size.height, size.width, CV_32FC1, const_cast<float*>(heights.data()));
  cv::Mat gray8, img;
  cv::normalize(gray, gray8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::applyColorMap(gray8, img, cv::COLORMAP_VIRIDIS);

  for (auto const& seg : segments)
    cv::line(img, cv::Point2f{seg.start.x, seg.start.y}, cv::Point2f{seg.end.x, seg.end.y}, cv::Scalar(0, 0, 0));

  cv::imwrite("plot.png", img);
}
} // namespace perf