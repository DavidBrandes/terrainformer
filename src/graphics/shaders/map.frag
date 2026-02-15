#version 330 core
in float height;
out vec4 FragColor;

void main() {
    vec3 color = mix(vec3(0.2, 0.3, 0.8), vec3(1.0, 1.0, 1.0), height);
    FragColor = vec4(color, 1.0);
}
