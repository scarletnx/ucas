#version 460
// points in (1 invocations), 4 points out with 8 vec4 each
layout(points, invocations = 1) in;
layout(points, max_vertices = 4) out;
layout(location = 0) in vec4 vin[];
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
layout(location = 5) out vec4 o5;
layout(location = 6) out vec4 o6;
layout(location = 7) out vec4 o7;
void main() {
    for (int i = 0; i < 4; i++) {
        gl_Position = vec4(vin[0].xy + float(i + gl_InvocationID) * 1e-3, 0.0, 1.0);
        o0 = vin[0] + 0.0;
        o1 = vin[0] + 1.0;
        o2 = vin[0] + 2.0;
        o3 = vin[0] + 3.0;
        o4 = vin[0] + 4.0;
        o5 = vin[0] + 5.0;
        o6 = vin[0] + 6.0;
        o7 = vin[0] + 7.0;
        EmitVertex();
    }
}
