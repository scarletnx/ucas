#version 460
// 1 vec4 inputs, 2 vec4 render targets
layout(location = 0) in vec4 i0;
layout(location = 0) out vec4 c0;
layout(location = 1) out vec4 c1;
void main() {
    vec4 t = i0;
    c0 = t * 0.25 + 0.0 / 8.0;
    c1 = t * 0.25 + 1.0 / 8.0;
}
