#include "utils/height_grid.h"

#include "utils/config.h"

#include <PerlinNoise.hpp>
#include <algorithm>
#include <cmath>

namespace {

template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

HeightGrid make_sinusoidal_height_grid(Size size, GridConfig::SinusoidalConfig config) {
  std::vector<float> positions;
  std::vector<float> heights;

  heights.reserve(static_cast<size_t>(size.width * size.height));
  positions.reserve(static_cast<size_t>(size.width * size.height * 2));

  float offset = HeightGrid::HEIGHT_RANGE.midpoint();
  float scale = config.amplitude * HeightGrid::HEIGHT_RANGE.span() / 2;
  float normalization_factor = std::min(static_cast<float>(size.width), static_cast<float>(size.height)) - 1;
  float factor = 2 * PI * config.frequency;

  for (int i = 0; i < size.height; ++i) {
    for (int j = 0; j < size.width; ++j) {
      float x = static_cast<float>(j);
      float y = static_cast<float>(i);

      float x_norm = (x / normalization_factor);
      float y_norm = (y / normalization_factor);

      float height = offset + scale * sinf(x_norm * factor) * cosf(y_norm * factor);
      height = std::clamp(height, HeightGrid::HEIGHT_RANGE.min, HeightGrid::HEIGHT_RANGE.max);

      positions.push_back(x);
      positions.push_back(y);
      heights.push_back(height);
    }
  }

  return HeightGrid{.size = size, .positions = positions, .heights = heights};
}

HeightGrid make_perlin_noise_height_grid(Size size, GridConfig::PerlinNoiseConfig config) {
  std::vector<float> positions;
  std::vector<float> heights;

  heights.reserve(static_cast<size_t>(size.width * size.height));
  positions.reserve(static_cast<size_t>(size.width * size.height * 2));

  siv::BasicPerlinNoise<float> perlin{config.seed};

  float normalization_factor = std::min(static_cast<float>(size.width), static_cast<float>(size.height)) - 1;
  float frequency = config.frequency / normalization_factor;

  float offset = HeightGrid::HEIGHT_RANGE.midpoint();
  float scale = HeightGrid::HEIGHT_RANGE.span() / 2;

  for (int i = 0; i < size.height; ++i) {
    for (int j = 0; j < size.width; ++j) {
      float x = static_cast<float>(j);
      float y = static_cast<float>(i);

      float height = perlin.normalizedOctave2D(x * frequency, y * frequency, config.octaves, config.persistence);
      height = offset + scale * height;

      positions.push_back(x);
      positions.push_back(y);
      heights.push_back(height);
    }
  }

  return HeightGrid{.size = size, .positions = positions, .heights = heights};
}

} // namespace

HeightGrid make_height_grid(GridConfig const& config) {
  Size size = config.size;

  return std::visit(
      Overloaded{[size](GridConfig::SinusoidalConfig config) { return make_sinusoidal_height_grid(size, config); },
                 [size](GridConfig::PerlinNoiseConfig config) { return make_perlin_noise_height_grid(size, config); }},
      config.initialization);
}