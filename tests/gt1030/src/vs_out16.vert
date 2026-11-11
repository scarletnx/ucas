#version 460
// 1 vec4 attributes in, 16 vec4 varyings out
layout(location = 0) in vec4 a0;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
layout(location = 5) out vec4 o5;
layout(location = 6) out vec4 o6;
layout(location = 7) out vec4 o7;
layout(location = 8) out vec4 o8;
layout(location = 9) out vec4 o9;
layout(location = 10) out vec4 o10;
layout(location = 11) out vec4 o11;
layout(location = 12) out vec4 o12;
layout(location = 13) out vec4 o13;
layout(location = 14) out vec4 o14;
layout(location = 15) out vec4 o15;
void main() {
    vec4 t = a0;
    gl_Position = vec4(t.xy, 0.0, 1.0);
    o0 = t + 0.0;
    o1 = t + 1.0;
    o2 = t + 2.0;
    o3 = t + 3.0;
    o4 = t + 4.0;
    o5 = t + 5.0;
    o6 = t + 6.0;
    o7 = t + 7.0;
    o8 = t + 8.0;
    o9 = t + 9.0;
    o10 = t + 10.0;
    o11 = t + 11.0;
    o12 = t + 12.0;
    o13 = t + 13.0;
    o14 = t + 14.0;
    o15 = t + 15.0;
}
