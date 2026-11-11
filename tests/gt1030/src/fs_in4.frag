#version 460
// 4 vec4 inputs, 1 vec4 render targets
layout(location = 0) in vec4 i0;
layout(location = 1) in vec4 i1;
layout(location = 2) in vec4 i2;
layout(location = 3) in vec4 i3;
layout(location = 0) out vec4 c0;
void main() {
    vec4 t = i0 + i1 + i2 + i3;
    c0 = t * 0.25 + 0.0 / 8.0;
}
