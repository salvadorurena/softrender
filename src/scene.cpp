#include "sr/scene.hpp"

#include <chrono>
#include <cmath>
#include <fstream>

namespace sr {

const char* view_mode_name(ViewMode m) {
    switch (m) {
        case ViewMode::Shaded: return "shaded";
        case ViewMode::Normals: return "normals";
        case ViewMode::Depth: return "depth";
        case ViewMode::UV: return "uv";
        case ViewMode::Mips: return "mips";
        default: return "?";
    }
}

bool parse_view_mode(const std::string& s, ViewMode& out) {
    for (int i = 0; i < int(ViewMode::Count); ++i)
        if (s == view_mode_name(ViewMode(i))) {
            out = ViewMode(i);
            return true;
        }
    return false;
}

Vec3 OrbitCamera::eye() const {
    return target + Vec3{std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)} * distance;
}

// ---------------------------------------------------------------------------
// Scene content

namespace {

float fract(float x) { return x - std::floor(x); }
float smoothstep(float a, float b, float x) {
    float t = clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

Texture make_floor_texture() {
    // One texture tile = one world unit: a 2x2 checker with soft grout lines.
    return Texture::generate(128, 128, [](float u, float v) {
        bool odd = (u < 0.5f) != (v < 0.5f);
        Vec3 c = odd ? Vec3(0.40f, 0.38f, 0.35f) : Vec3(0.23f, 0.22f, 0.21f);
        float edge = std::min({u, 1.0f - u, v, 1.0f - v, std::abs(u - 0.5f), std::abs(v - 0.5f)});
        return lerp(Vec3(0.10f, 0.10f, 0.10f), c, smoothstep(0.0f, 0.02f, edge));
    });
}

Texture make_knot_texture() {
    // Bands along the knot with a thin gold seam between them.
    return Texture::generate(512, 64, [](float u, float v) {
        float band = fract(u * 8.0f);
        Vec3 a{0.80f, 0.18f, 0.08f}, b{0.03f, 0.22f, 0.28f};
        Vec3 c = band < 0.5f ? a : b;
        float seam = std::min(std::abs(band - 0.5f), std::min(band, 1.0f - band));
        c = lerp(Vec3(0.95f, 0.65f, 0.20f), c, smoothstep(0.015f, 0.035f, seam));
        // Faint ridges around the tube.
        return c * (0.85f + 0.15f * std::cos(v * 2.0f * kPi * 6.0f));
    });
}

Texture make_uv_checker() {
    return Texture::generate(256, 256, [](float u, float v) {
        bool odd = (int(u * 8) + int(v * 8)) % 2;
        return odd ? Vec3(0.75f, 0.72f, 0.68f) : Vec3(0.30f, 0.29f, 0.28f);
    });
}

}  // namespace

Scene Scene::create(const std::string& obj_path) {
    Scene s;
    if (obj_path.empty()) {
        std::string bundled = std::string(SR_ASSET_DIR) + "/torus_knot.obj";
        s.model = std::ifstream(bundled) ? load_obj(bundled) : make_torus_knot(2, 3, 1.0f, 0.28f, 256, 32);
        s.model_texture = make_knot_texture();
        s.model_lift = 0.45f;
    } else {
        s.model = load_obj(obj_path);
        bool has_uvs = false;
        for (const Vertex& v : s.model.vertices) has_uvs |= (v.uv.x != 0.0f || v.uv.y != 0.0f);
        s.model_texture = has_uvs ? make_uv_checker() : Texture(1, 1, Vec3(0.70f, 0.66f, 0.60f));
    }
    s.model.normalize_to(2.6f);
    Vec3 lo, hi;
    s.model.bounds(lo, hi);
    for (Vertex& v : s.model.vertices) v.position.y -= lo.y;  // rest on y = 0
    s.focus = {0.0f, s.model_lift + (hi.y - lo.y) * 0.45f, 0.0f};

    s.ground = make_plane(60.0f, 60.0f);
    s.ground_texture = make_floor_texture();
    return s;
}

Mat4 Scene::model_matrix(float spin) const { return Mat4::translate({0, model_lift, 0}) * Mat4::rotate_y(spin); }

// ---------------------------------------------------------------------------
// Shaders

namespace {

// Interpolated surface attributes shared by all of the main-pass shaders.
struct Surface {
    Vec3 world;
    Vec3 normal;
    Vec2 uv;
    float view_depth;

    Surface operator+(const Surface& o) const {
        return {world + o.world, normal + o.normal, uv + o.uv, view_depth + o.view_depth};
    }
    Surface operator*(float s) const { return {world * s, normal * s, uv * s, view_depth * s}; }
};

struct SurfaceVertexShader {
    using Varying = Surface;
    Mat4 model, view, view_proj;

    Vec4 vertex(const Vertex& v, Surface& out) const {
        Vec3 world = model.transform_point(v.position);
        out.world = world;
        out.normal = model.transform_dir(v.normal);
        out.uv = v.uv;
        out.view_depth = -view.transform_point(world).z;
        return view_proj * Vec4(world, 1.0f);
    }
};

struct DepthOnlyShader {
    struct Varying {
        Varying operator+(const Varying&) const { return {}; }
        Varying operator*(float) const { return {}; }
    };
    Mat4 mvp;
    Vec4 vertex(const Vertex& v, Varying&) const { return mvp * Vec4(v.position, 1.0f); }
    bool fragment(const Varying&, Vec3&) const { return true; }
};

// Shadow map lookup with 4x4 percentage-closer filtering.
struct ShadowLookup {
    const Framebuffer* map = nullptr;
    Mat4 light_view_proj;

    float visibility(Vec3 world, float n_dot_l) const {
        if (!map) return 1.0f;
        Vec4 p = light_view_proj * Vec4(world, 1.0f);  // orthographic: w == 1
        const int size = map->width();
        float tx = (p.x * 0.5f + 0.5f) * size, ty = (0.5f - p.y * 0.5f) * size;
        float depth = p.z * 0.5f + 0.5f;
        if (depth >= 1.0f) return 1.0f;
        // Slope-scaled bias: surfaces at grazing angles to the light need more.
        float bias = 0.0004f + 0.0016f * (1.0f - n_dot_l);
        float lit = 0.0f;
        for (int dy = -2; dy < 2; ++dy)
            for (int dx = -2; dx < 2; ++dx) {
                int x = int(std::floor(tx + dx + 0.5f)), y = int(std::floor(ty + dy + 0.5f));
                if (x < 0 || y < 0 || x >= size || y >= size || depth - bias <= map->depth_at(x, y)) lit += 1.0f;
            }
        return lit * (1.0f / 16.0f);
    }
};

Vec3 aces_tonemap(Vec3 x) {
    auto f = [](float v) { return clamp01((v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f)); };
    return {f(x.x), f(x.y), f(x.z)};
}

const Vec3 kSkyZenith{0.05f, 0.09f, 0.20f};
const Vec3 kSkyHorizon{0.36f, 0.39f, 0.45f};

struct LitShader : SurfaceVertexShader {
    const Texture* albedo = nullptr;
    bool use_texture = true;
    TextureFilter filter = TextureFilter::Anisotropic;
    Vec3 eye, light_dir, sun_color{2.3f, 2.15f, 1.9f};
    float shininess = 48.0f, specular = 0.35f;
    ShadowLookup shadow;

    bool fragment(const Surface& in, const Derivatives<Surface>& d, Vec3& out) const {
        Vec3 n = normalize(in.normal);
        Vec3 v = normalize(eye - in.world);
        if (dot(n, v) < 0) n = -n;  // shade back faces of open meshes sensibly

        Vec3 base = use_texture ? albedo->sample(in.uv, d.ddx.uv, d.ddy.uv, filter) : Vec3(0.7f);
        float n_dot_l = std::max(dot(n, light_dir), 0.0f);
        float vis = n_dot_l > 0 ? shadow.visibility(in.world, n_dot_l) : 0.0f;

        // Blinn-Phong direct light + hemispheric ambient (sky above, bounce below).
        Vec3 h = normalize(light_dir + v);
        float spec = n_dot_l > 0 ? std::pow(std::max(dot(n, h), 0.0f), shininess) * specular : 0.0f;
        Vec3 ambient = lerp(Vec3(0.10f, 0.09f, 0.08f), Vec3(0.30f, 0.36f, 0.48f), n.y * 0.5f + 0.5f);
        Vec3 color = base * (ambient + sun_color * (n_dot_l * vis)) + sun_color * (spec * vis);

        // Distance fog into the horizon color hides the edge of the ground plane.
        float fog = smoothstep(10.0f, 26.0f, in.view_depth);
        out = lerp(aces_tonemap(color), kSkyHorizon, fog);
        return true;
    }
};

// Debug shaders output display-referred values; converting them to linear here
// cancels the framebuffer's sRGB encode, so the stored byte equals the value.
float srgb_to_linear(float s) {
    s = clamp01(s);
    return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
}
Vec3 display(Vec3 c) { return {srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z)}; }

struct NormalShader : SurfaceVertexShader {
    bool fragment(const Surface& in, Vec3& out) const {
        out = display(normalize(in.normal) * 0.5f + Vec3(0.5f));
        return true;
    }
};

struct DepthShader : SurfaceVertexShader {
    float near_vis = 0, far_vis = 1;
    bool fragment(const Surface& in, Vec3& out) const {
        float t = clamp01((in.view_depth - near_vis) / (far_vis - near_vis));
        out = display(Vec3((1.0f - t) * (1.0f - t)));
        return true;
    }
};

// UV gradient with tile borders drawn as anti-aliased lines of constant 1.5 px
// width, using derivatives the same way GLSL's fwidth() would.
struct UVShader : SurfaceVertexShader {
    bool fragment(const Surface& in, const Derivatives<Surface>& d, Vec3& out) const {
        float u = fract(in.uv.x), v = fract(in.uv.y);
        float fu = std::abs(d.ddx.uv.x) + std::abs(d.ddy.uv.x), fv = std::abs(d.ddx.uv.y) + std::abs(d.ddy.uv.y);
        float du = std::min(u, 1.0f - u) / std::max(fu, 1e-6f), dv = std::min(v, 1.0f - v) / std::max(fv, 1e-6f);
        float line = 1.0f - 0.55f * (1.0f - smoothstep(0.0f, 1.5f, std::min(du, dv)));
        out = display(Vec3(u, v, 0.25f) * line);
        return true;
    }
};

// Colors each pixel by the mip level trilinear filtering would pick:
// blue = full resolution, then cyan, green, yellow, orange, red, magenta as it shrinks.
struct MipShader : SurfaceVertexShader {
    const Texture* albedo = nullptr;
    bool fragment(const Surface& in, const Derivatives<Surface>& d, Vec3& out) const {
        static const Vec3 palette[] = {{0.15f, 0.35f, 1.0f}, {0.1f, 0.85f, 0.95f}, {0.2f, 0.9f, 0.3f},
                                       {0.95f, 0.9f, 0.2f},  {1.0f, 0.55f, 0.1f},  {0.95f, 0.15f, 0.1f},
                                       {0.85f, 0.2f, 0.9f}};
        constexpr int n = int(sizeof(palette) / sizeof(palette[0]));
        float lod = std::clamp(albedo->lod(d.ddx.uv, d.ddy.uv), 0.0f, float(n - 1));
        int i = std::min(int(lod), n - 2);
        Vec3 tint = lerp(palette[i], palette[i + 1], lod - float(i));
        Vec3 tex = albedo->sample(in.uv, d.ddx.uv, d.ddy.uv, TextureFilter::Trilinear);
        float lum = std::clamp(dot(tex, Vec3(0.3f, 0.6f, 0.1f)) * 2.0f, 0.0f, 1.0f);
        out = display(tint * (0.55f + 0.45f * lum));
        return true;
    }
};

// Sky gradient by view-ray elevation (per row; the orbit camera never rolls).
void clear_sky(Framebuffer& fb, const OrbitCamera& cam) {
    fb.clear_depth();
    const float tan_half = std::tan(cam.fovy * 0.5f);
    Vec3 forward = normalize(cam.target - cam.eye());
    float cam_elev = std::asin(std::clamp(forward.y, -1.0f, 1.0f));
    for (int y = 0; y < fb.height(); ++y) {
        float ndc_y = 1.0f - 2.0f * (y + 0.5f) / fb.height();
        float elev = cam_elev + std::atan(ndc_y * tan_half);
        Vec3 c = elev > 0 ? lerp(kSkyHorizon, kSkyZenith, std::pow(clamp01(elev / (0.5f * kPi)), 0.6f)) : kSkyHorizon;
        uint32_t packed = pack_rgba(c);
        for (int x = 0; x < fb.width(); ++x) fb.pixel(x, y) = packed;
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

FrameStats Renderer::render(Framebuffer& fb, const Scene& scene, const OrbitCamera& cam, const RenderSettings& rs) {
    FrameStats stats;
    const Mat4 model = scene.model_matrix(rs.model_spin);
    const Mat4 ground = Mat4::identity();
    const Mat4 view = cam.view();
    const Mat4 view_proj = cam.projection(float(fb.width()) / float(fb.height())) * view;
    const std::span<const Vertex> model_verts(scene.model.vertices), ground_verts(scene.ground.vertices);
    const std::span<const uint32_t> model_idx(scene.model.indices), ground_idx(scene.ground.indices);

    // --- Pass 1: shadow map (orthographic, from the sun) ----------------------
    Vec3 light_dir = normalize({std::cos(rs.light_pitch) * std::sin(rs.light_yaw), std::sin(rs.light_pitch),
                                std::cos(rs.light_pitch) * std::cos(rs.light_yaw)});
    const float radius = 4.0f;
    Mat4 light_vp = Mat4::ortho(-radius, radius, -radius, radius, 1.0f, 40.0f) *
                    Mat4::look_at(scene.focus + light_dir * 20.0f, scene.focus, {0, 1, 0});
    const bool shadows = rs.shadows && rs.mode == ViewMode::Shaded;
    auto t0 = std::chrono::steady_clock::now();
    if (shadows) {
        shadow_map_.clear_depth();
        DepthOnlyShader depth{light_vp * model};
        // Rendering back faces into the shadow map pushes acne off the lit side.
        draw(shadow_map_, depth, model_verts, model_idx, {.cull = CullMode::Front});
        stats.shadow_ms = ms_since(t0);
    }

    // --- Pass 2: main view ------------------------------------------------------
    t0 = std::chrono::steady_clock::now();
    clear_sky(fb, cam);
    auto draw_both = [&](auto shader, const Texture* model_tex) {
        constexpr bool kTextured = requires(decltype(shader)& s) { s.albedo; };
        shader.view = view;
        shader.view_proj = view_proj;
        shader.model = ground;
        if constexpr (kTextured) shader.albedo = &scene.ground_texture;
        stats.ground = draw(fb, shader, ground_verts, ground_idx);
        shader.model = model;
        if constexpr (kTextured) shader.albedo = model_tex;
        stats.model = draw(fb, shader, model_verts, model_idx);
    };
    switch (rs.mode) {
        case ViewMode::Shaded: {
            LitShader s;
            s.eye = cam.eye();
            s.light_dir = light_dir;
            s.use_texture = rs.textures;
            s.filter = rs.filter;
            if (shadows) s.shadow = {&shadow_map_, light_vp};
            draw_both(s, &scene.model_texture);
            break;
        }
        case ViewMode::Normals: draw_both(NormalShader{}, nullptr); break;
        case ViewMode::Depth: {
            DepthShader s;
            s.near_vis = std::max(0.0f, cam.distance - 3.0f);
            s.far_vis = cam.distance + 10.0f;
            draw_both(s, nullptr);
            break;
        }
        case ViewMode::Mips: draw_both(MipShader{}, &scene.model_texture); break;
        default: draw_both(UVShader{}, nullptr); break;
    }
    stats.main_ms = ms_since(t0);

    // --- Pass 3: wireframe overlay ---------------------------------------------
    if (rs.wireframe) {
        t0 = std::chrono::steady_clock::now();
        Vec3 color = rs.mode == ViewMode::Depth ? Vec3(1.0f, 0.5f, 0.1f) : Vec3(0.02f, 0.02f, 0.025f);
        draw_wireframe(fb, scene.model, view_proj * model, color);
        stats.wire_ms = ms_since(t0);
    }
    return stats;
}

// ---------------------------------------------------------------------------
// Wireframe

void draw_wireframe(Framebuffer& fb, const Mesh& mesh, const Mat4& mvp, Vec3 color) {
    const int W = fb.width(), H = fb.height();
    const uint32_t packed = pack_rgba(color);
    const float depth_bias = 2e-4f;

    struct P { float x, y, z; bool ok; };
    std::vector<P> screen(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        Vec4 c = mvp * Vec4(mesh.vertices[i].position, 1.0f);
        bool ok = c.w > 1e-4f && c.z >= -c.w && c.z <= c.w;  // whole triangles behind the near plane are dropped
        float iw = ok ? 1.0f / c.w : 0.0f;
        screen[i] = {(c.x * iw * 0.5f + 0.5f) * W, (0.5f - c.y * iw * 0.5f) * H, c.z * iw * 0.5f + 0.5f, ok};
    }

    auto line = [&](const P& a, const P& b) {
        float dx = b.x - a.x, dy = b.y - a.y;
        int steps = int(std::ceil(std::max(std::abs(dx), std::abs(dy))));
        if (steps > 4 * (W + H)) return;  // degenerate / way off-screen
        float inv = steps ? 1.0f / steps : 0.0f;
        for (int i = 0; i <= steps; ++i) {
            float t = i * inv;
            int x = int(a.x + dx * t), y = int(a.y + dy * t);
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            float z = a.z + (b.z - a.z) * t;
            if (z <= fb.depth_at(x, y) + depth_bias) fb.pixel(x, y) = packed;
        }
    };

    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const P& a = screen[mesh.indices[i]];
        const P& b = screen[mesh.indices[i + 1]];
        const P& c = screen[mesh.indices[i + 2]];
        if (!a.ok || !b.ok || !c.ok) continue;
        float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (area >= 0) continue;  // back-facing (see the winding note in rasterizer.hpp)
        line(a, b);
        line(b, c);
        line(c, a);
    }
}

}  // namespace sr
