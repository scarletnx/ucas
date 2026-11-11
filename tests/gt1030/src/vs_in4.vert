#version 460
// 4 vec4 attributes in, 1 vec4 varyings out
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 3) in vec4 a3;
layout(location = 0) out vec4 o0;
void main() {
    vec4 t = a0 + a1 + a2 + a3;
    gl_Position = vec4(t.xy, 0.0, 1.0);
    o0 = t + 0.0;
}
