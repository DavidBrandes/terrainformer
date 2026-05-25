#pragma once

constexpr float PI = 3.14159265358979323846f;

struct Point {
  float x;
  float y;
};

struct Size {
  int width;
  int height;

  bool empty() const { return width <= 0 || height <= 0; }
};

struct Range {
  float min;
  float max;

  constexpr float span() const { return max - min; }
  constexpr float midpoint() const { return (max + min) / 2.0f; }
};

struct Circle {
  Point center;
  float radius;
};

struct BrushDab {
  Circle circle;
  float intensity;
};