#version 330 core
out vec4 FragColor;
uniform float t;

// clang-format off
const vec3 low  = vec3(46 / 255.0, 20 / 255.0, 55 / 255.0);
const vec3 high = vec3(148 / 255.0, 142 / 255.0, 153 / 255.0);
// clang-format on

void main() {
  vec3 color = mix(low, high, t);
  FragColor = vec4(color, 1.0);
}
