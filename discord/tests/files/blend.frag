#version 450
layout(binding = 0) uniform sampler2D a;
layout(binding = 1) uniform sampler2D b;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() { o = mix(texture(a, uv), texture(b, uv), uv.x); }
