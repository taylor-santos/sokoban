#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

// Instanced data
layout(location = 3) in mat4 transform_instance;

layout(location = 0) out vec2 v_texcoord;
layout(location = 1) out vec3 v_view_dir;
layout(location = 2) out vec3 v_world_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
} uniforms;

layout(set = 2, binding = 0) uniform Object {
    mat4 transform;
} object;

void main()
{
    vec4 worldPos = transform_instance * object.transform * vec4(position, 1.0);
    gl_Position = uniforms.proj * uniforms.view * worldPos;

    v_texcoord = texcoord;
    v_view_dir = uniforms.camera_pos - worldPos.xyz;
    v_world_normal = mat3(transform_instance * object.transform) * normal;
}
