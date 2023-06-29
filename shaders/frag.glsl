#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

const vec3 LIGHT   = vec3(0.0, 0.0, 1.0);
const vec3 AMBIENT = vec3(0.0, 0.68, 1.0);

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));

    vec2 uv = vec2(v_texcoord.x, 1-v_texcoord.y);
    vec3 albedo = texture(tex, uv).rgb;

    f_color = vec4(mix(albedo * AMBIENT * 0.6, albedo, brightness), 1.0);
}
