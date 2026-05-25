#version 330 core
in float height;
out vec4 FragColor;

// clang-format off
const vec3 low  = vec3(8 / 255.0, 80 / 255.0, 120 / 255.0);
const vec3 high = vec3(133 / 255.0, 216 / 255.0, 206 / 255.0);
// clang-format on

void main() {
  vec3 color = mix(low, high, height);
  FragColor = vec4(color, 1.0);
}