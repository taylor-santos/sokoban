#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_texcoord;
layout(location = 2) out vec3 v_view_dir;
layout(location = 3) out vec3 v_world_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(set = 2, binding = 0) uniform Object {
    mat4 transform;
} object;

void main() {
    mat4 worldview = uniforms.view * uniforms.world * object.transform;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    v_world_normal = transpose(inverse(mat3(uniforms.world))) * normal;
    v_texcoord = texcoord;
    v_view_dir = normalize(vec3(inverse(uniforms.view) * vec4(0.0, 0.0, 0.0, 1.0) - uniforms.world * vec4(position, 1.0)));
}
