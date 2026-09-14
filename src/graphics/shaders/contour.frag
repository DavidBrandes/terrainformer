#version 330 core

out vec4 FragColor;

// clang-format off
const vec3 color = vec3(92 / 255.0, 98 / 255.0, 95 / 255.0);
// clang-format on

void main() { FragColor = vec4(color, 1.0); }