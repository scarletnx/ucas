#version 460
// points in (1 invocations), 8 points out with 1 vec4 each
layout(points, invocations = 1) in;
layout(points, max_vertices = 8) out;
layout(location = 0) in vec4 vin[];
layout(location = 0) out vec4 o0;
void main() {
    for (int i = 0; i < 8; i++) {
        gl_Position = vec4(vin[0].xy + float(i + gl_InvocationID) * 1e-3, 0.0, 1.0);
        o0 = vin[0] + 0.0;
        EmitVertex();
    }
}
