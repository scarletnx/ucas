#version 460
// 1 vec4 attributes in, 2 vec4 varyings out
layout(location = 0) in vec4 a0;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    vec4 t = a0;
    gl_Position = vec4(t.xy, 0.0, 1.0);
    o0 = t + 0.0;
    o1 = t + 1.0;
}
