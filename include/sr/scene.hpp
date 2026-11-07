// Everything above the raw pipeline: scene content, camera, shaders and the
// multi-pass frame (shadow map -> main pass -> wireframe overlay).
// Shared by the headless renderer and the interactive viewer.
#pragma once

#include <string>

#include "sr/framebuffer.hpp"
#include "sr/mesh.hpp"
#include "sr/rasterizer.hpp"
#include "sr/texture.hpp"

namespace sr {

enum class ViewMode { Shaded, Normals, Depth, UV, Mips, Count };
const char* view_mode_name(ViewMode m);
bool parse_view_mode(const std::string& s, ViewMode& out);

struct OrbitCamera {
    Vec3 target{0.0f, 1.0f, 0.0f};
    float yaw = radians(35.0f);
    float pitch = radians(22.0f);
    float distance = 6.5f;
    float fovy = radians(45.0f);
    float znear = 0.1f, zfar = 60.0f;

    Vec3 eye() const;
    Mat4 view() const { return Mat4::look_at(eye(), target, {0, 1, 0}); }
    Mat4 projection(float aspect) const { return Mat4::perspective(fovy, aspect, znear, zfar); }
};

struct RenderSettings {
    ViewMode mode = ViewMode::Shaded;
    bool wireframe = false;
    bool shadows = true;
    bool textures = true;
    TextureFilter filter = TextureFilter::Anisotropic;
    float light_yaw = radians(-40.0f);
    float light_pitch = radians(55.0f);
    float model_spin = 0.0f;  // radians around Y
};

struct Scene {
    Mesh model;
    Mesh ground;
    Texture model_texture;
    Texture ground_texture;
    float model_lift = 0.0f;  // model's bounding-box bottom sits this far above the ground
    Vec3 focus;               // a good camera / shadow-map target

    // Loads `obj_path`, or the bundled torus knot when empty. Throws on load failure.
    static Scene create(const std::string& obj_path = "");
    Mat4 model_matrix(float spin) const;
};

struct FrameStats {
    RasterStats model, ground;
    double shadow_ms = 0, main_ms = 0, wire_ms = 0;
    double total_ms() const { return shadow_ms + main_ms + wire_ms; }
};

class Renderer {
public:
    explicit Renderer(int shadow_map_size = 2048) : shadow_map_(shadow_map_size, shadow_map_size, false) {}
    FrameStats render(Framebuffer& target, const Scene& scene, const OrbitCamera& camera,
                      const RenderSettings& settings);

private:
    Framebuffer shadow_map_;  // depth only
};

// Depth-tested line rendering of a mesh's front-facing edges (classic DDA).
void draw_wireframe(Framebuffer& fb, const Mesh& mesh, const Mat4& mvp, Vec3 color);

}  // namespace sr
