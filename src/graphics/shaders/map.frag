#version 330 core
in float height;
out vec4 FragColor;

// clang-format off
const vec3 low  = vec3(171 / 255.0, 186 / 255.0, 171 / 255.0);
const vec3 high = vec3(255 / 255.0, 255 / 255.0, 255 / 255.0);
// clang-format on

void main() {
  vec3 color = mix(low, high, height);
  FragColor = vec4(color, 1.0);
}