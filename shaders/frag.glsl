#version 450

layout(location = 0) in vec2 v_texcoord;
layout(location = 1) in vec3 v_view_dir;
layout(location = 2) in vec3 v_world_normal;

layout(location = 0) out vec4 f_color;

const vec3 LIGHT   = vec3(0.0, 1.0, -1.0);

layout(set = 1, binding = 0) uniform sampler2D tex;

const float PI = 3.14159265358979323846;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    vec2 uv = vec2(v_texcoord.x, v_texcoord.y);
    vec4 albedo = texture(tex, uv);
    if (albedo.a < 1.0) {
        if (albedo.a < 0.01) discard;
        float r = random(uv);
        if (albedo.a * albedo.a < r) {
            discard;
        }
    }

    float specular;
    vec3 normal_dir = normalize(v_world_normal);
    vec3 view_dir = normalize(v_view_dir);
    vec3 light_dir = normalize(LIGHT);
    if (dot(normal_dir, LIGHT) < 0.0) {
        specular = 0.0;
    } else {
        specular = pow(max(0.0, dot(reflect(-light_dir, normal_dir), view_dir)), 10.0);
    }
    float diffuse = max(0.0, dot(normal_dir, light_dir));

    vec3 color = albedo.rgb * (diffuse + specular + 0.2);

    f_color = vec4(color, 1.0);
}
