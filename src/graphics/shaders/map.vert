#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aHeight;
out float height;

uniform vec2 cropMin;
uniform vec2 cropMax;
uniform float minValue;
uniform float maxValue;

void main() {
    height = (aHeight - minValue) / (maxValue - minValue);

    vec2 normalized;
    normalized.x = ((aPos.x - cropMin.x) / (cropMax.x - cropMin.x)) * 2.0 - 1.0;
    normalized.y = 1.0 - ((aPos.y - cropMin.y) / (cropMax.y - cropMin.y)) * 2.0;

    gl_Position = vec4(normalized, 0.0, 1.0);
}
