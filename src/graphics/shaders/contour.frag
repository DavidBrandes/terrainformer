#version 330 core

out vec4 FragColor;

// clang-format off
const vec3 color = vec3(
    46.0 / 255.0,
    20.0 / 255.0,
    55.0 / 255.0
);
// clang-format on

void main() { FragColor = vec4(color, 1.0); }