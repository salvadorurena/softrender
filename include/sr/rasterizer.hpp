// The heart of the renderer: a programmable, multithreaded triangle pipeline.
//
//   vertex shader -> primitive assembly -> homogeneous clipping -> perspective divide
//   -> viewport + backface culling -> fixed-point edge-function rasterization
//   -> perspective-correct interpolation -> early depth test -> fragment shader
//
// A shader is any type that looks like:
//
//   struct MyShader {
//       struct Varying { ...; Varying operator+(const Varying&) const;
//                             Varying operator*(float) const; };
//       Vec4 vertex(const VertexT& in, Varying& out) const;     // returns clip-space position
//       bool fragment(const Varying& in, Vec3& color) const;   // false = discard
//   };
//
// Shaders are template parameters, so the whole pipeline is inlined per shader.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include "sr/framebuffer.hpp"
#include "sr/math.hpp"

namespace sr {

enum class CullMode { None, Back, Front };

struct DrawState {
    CullMode cull = CullMode::Back;
    bool depth_test = true;
    bool depth_write = true;
};

struct RasterStats {
    size_t triangles_in = 0;
    size_t triangles_rasterized = 0;  // after clipping and culling
};

// ---------------------------------------------------------------------------
// Tiny fork/join helper. Spawning ~10 threads per draw costs a few microseconds,
// negligible next to rasterization, and keeps the code free of a thread pool.
inline unsigned worker_count() {
    static const unsigned n = std::max(1u, std::thread::hardware_concurrency());
    return n;
}

template <class F>
void parallel_for(unsigned n, F&& fn) {
    std::vector<std::thread> threads;
    threads.reserve(n ? n - 1 : 0);
    for (unsigned i = 1; i < n; ++i) threads.emplace_back([&fn, i] { fn(i); });
    if (n) fn(0u);
    for (auto& t : threads) t.join();
}

namespace detail {

constexpr int kSubpixelBits = 8;  // 1/256 pixel vertex precision
constexpr int kSubpixel = 1 << kSubpixelBits;
constexpr float kGuardBand = 4.0f;  // clip x/y only beyond 4x the viewport extents
constexpr int kBandHeight = 8;      // rows per interleaved work band

template <class V>
struct ClipVertex {
    Vec4 pos;
    V var;
};

template <class V>
struct ScreenTri {
    int64_t x[3], y[3];  // fixed-point screen coordinates
    float z[3];          // depth in [0, 1]
    float inv_w[3];      // 1/w for perspective-correct interpolation
    V var[3];            // varyings pre-multiplied by 1/w
    int min_x, min_y, max_x, max_y;
    int64_t area;
};

// Signed distance of a clip-space point to each frustum plane (>= 0 is inside).
inline float plane_dist(const Vec4& p, int plane) {
    switch (plane) {
        case 0: return p.z + p.w;               // near
        case 1: return p.w - p.z;               // far
        case 2: return p.x + kGuardBand * p.w;  // left
        case 3: return kGuardBand * p.w - p.x;  // right
        case 4: return p.y + kGuardBand * p.w;  // bottom
        default: return kGuardBand * p.w - p.y; // top
    }
}

// Sutherland-Hodgman clipping of a polygon against the planes in `mask`.
// Returns the vertex count (0 if fully clipped). `poly` must hold up to 9 vertices.
template <class V>
int clip_polygon(ClipVertex<V>* poly, int count, unsigned mask) {
    ClipVertex<V> tmp[9];
    for (int plane = 0; plane < 6; ++plane) {
        if (!(mask & (1u << plane))) continue;
        int out = 0;
        for (int i = 0; i < count; ++i) {
            const ClipVertex<V>& a = poly[i];
            const ClipVertex<V>& b = poly[(i + 1) % count];
            float da = plane_dist(a.pos, plane), db = plane_dist(b.pos, plane);
            if (da >= 0) tmp[out++] = a;
            if ((da >= 0) != (db >= 0)) {
                float t = da / (da - db);
                tmp[out++] = {a.pos + (b.pos - a.pos) * t, a.var + (b.var * t + a.var * -t)};
            }
        }
        count = out;
        std::copy(tmp, tmp + count, poly);
        if (count < 3) return 0;
    }
    return count;
}

// Top-left fill rule: pixels exactly on an edge belong to exactly one of the two
// triangles sharing it. Screen space is y-down and triangles are wound so area > 0.
inline bool is_top_left(int64_t ax, int64_t ay, int64_t bx, int64_t by) {
    int64_t dx = bx - ax, dy = by - ay;
    return (dy == 0 && dx > 0) || dy < 0;
}

}  // namespace detail

template <class Shader, class VertexT>
RasterStats draw(Framebuffer& fb, const Shader& shader, std::span<const VertexT> vertices,
                 std::span<const uint32_t> indices, const DrawState& state = {}) {
    using V = typename Shader::Varying;
    using namespace detail;

    const unsigned workers = worker_count();
    const int W = fb.width(), H = fb.height();
    RasterStats stats;
    stats.triangles_in = indices.size() / 3;

    // 1. Vertex shading (each vertex once, shared by all triangles that index it).
    std::vector<ClipVertex<V>> clip(vertices.size());
    parallel_for(workers, [&](unsigned t) {
        size_t begin = vertices.size() * t / workers, end = vertices.size() * (t + 1) / workers;
        for (size_t i = begin; i < end; ++i) clip[i].pos = shader.vertex(vertices[i], clip[i].var);
    });

    // 2. Primitive assembly, clipping, projection, culling and setup.
    std::vector<std::vector<ScreenTri<V>>> bins(workers);
    const size_t tri_count = indices.size() / 3;
    parallel_for(workers, [&](unsigned t) {
        size_t begin = tri_count * t / workers, end = tri_count * (t + 1) / workers;
        auto& out = bins[t];
        out.reserve(end - begin);

        auto emit = [&](const ClipVertex<V>& a, const ClipVertex<V>& b, const ClipVertex<V>& c) {
            const ClipVertex<V>* v[3] = {&a, &b, &c};
            ScreenTri<V> tri;
            float fx[3], fy[3];
            for (int k = 0; k < 3; ++k) {
                float iw = 1.0f / v[k]->pos.w;
                fx[k] = (v[k]->pos.x * iw * 0.5f + 0.5f) * float(W);
                fy[k] = (0.5f - v[k]->pos.y * iw * 0.5f) * float(H);  // NDC y-up -> screen y-down
                tri.x[k] = int64_t(std::llround(fx[k] * kSubpixel));
                tri.y[k] = int64_t(std::llround(fy[k] * kSubpixel));
                tri.z[k] = v[k]->pos.z * iw * 0.5f + 0.5f;
                tri.inv_w[k] = iw;
                tri.var[k] = v[k]->var * iw;
            }
            int64_t area = (tri.x[1] - tri.x[0]) * (tri.y[2] - tri.y[0]) -
                           (tri.y[1] - tri.y[0]) * (tri.x[2] - tri.x[0]);
            if (area == 0) return;
            // Counter-clockwise in NDC (front-facing) becomes clockwise once y is
            // flipped to point down, which makes this cross product negative.
            bool front = area < 0;
            if ((state.cull == CullMode::Back && !front) || (state.cull == CullMode::Front && front)) return;
            if (area < 0) {  // rewind so the rasterizer only handles one orientation
                std::swap(tri.x[1], tri.x[2]); std::swap(tri.y[1], tri.y[2]);
                std::swap(tri.z[1], tri.z[2]); std::swap(tri.inv_w[1], tri.inv_w[2]);
                std::swap(tri.var[1], tri.var[2]);
                area = -area;
            }
            tri.area = area;
            // Pixel bounding box (pixel centers sit at +0.5), clamped to the viewport.
            auto lo = [](int64_t a, int64_t b, int64_t c) { return std::min({a, b, c}); };
            auto hi = [](int64_t a, int64_t b, int64_t c) { return std::max({a, b, c}); };
            tri.min_x = std::max(0, int((lo(tri.x[0], tri.x[1], tri.x[2]) - kSubpixel / 2) >> kSubpixelBits));
            tri.min_y = std::max(0, int((lo(tri.y[0], tri.y[1], tri.y[2]) - kSubpixel / 2) >> kSubpixelBits));
            tri.max_x = std::min(W - 1, int((hi(tri.x[0], tri.x[1], tri.x[2]) - kSubpixel / 2) >> kSubpixelBits) + 1);
            tri.max_y = std::min(H - 1, int((hi(tri.y[0], tri.y[1], tri.y[2]) - kSubpixel / 2) >> kSubpixelBits) + 1);
            if (tri.min_x > tri.max_x || tri.min_y > tri.max_y) return;
            out.push_back(tri);
        };

        for (size_t i = begin; i < end; ++i) {
            ClipVertex<V> poly[9] = {clip[indices[3 * i]], clip[indices[3 * i + 1]], clip[indices[3 * i + 2]]};
            // Outcodes: trivially reject if all three are outside one plane,
            // and only run the clipper against planes that are actually crossed.
            unsigned any_out = 0, all_out = 0x3F;
            for (int k = 0; k < 3; ++k) {
                unsigned code = 0;
                for (int p = 0; p < 6; ++p)
                    if (plane_dist(poly[k].pos, p) < 0) code |= 1u << p;
                any_out |= code;
                all_out &= code;
            }
            if (all_out) continue;
            int n = any_out ? clip_polygon(poly, 3, any_out) : 3;
            for (int k = 1; k + 1 < n; ++k) emit(poly[0], poly[k], poly[k + 1]);
        }
    });
    for (auto& b : bins) stats.triangles_rasterized += b.size();

    // 3. Rasterization. The screen is split into 8-row bands dealt round-robin to
    //    threads, so every pixel is owned by exactly one thread: no locks, no races,
    //    and triangle submission order is preserved within each pixel.
    const bool write_color = fb.has_color();
    parallel_for(workers, [&](unsigned t) {
        for (const auto& bin : bins) {
            for (const ScreenTri<V>& tri : bin) {
                int first_band = tri.min_y / kBandHeight, last_band = tri.max_y / kBandHeight;
                // Edge function E_ab(p) = A*px + B*py + C, evaluated in fixed point.
                int64_t A[3], B[3], C[3], bias[3];
                for (int e = 0; e < 3; ++e) {
                    int a = (e + 1) % 3, b = (e + 2) % 3;  // edge opposite vertex e
                    A[e] = -(tri.y[b] - tri.y[a]);
                    B[e] = tri.x[b] - tri.x[a];
                    C[e] = -(A[e] * tri.x[a] + B[e] * tri.y[a]);
                    bias[e] = is_top_left(tri.x[a], tri.y[a], tri.x[b], tri.y[b]) ? 0 : -1;
                }
                const float inv_area = 1.0f / float(tri.area);

                for (int band = first_band; band <= last_band; ++band) {
                    if (unsigned(band) % workers != t) continue;
                    int y0 = std::max(tri.min_y, band * kBandHeight);
                    int y1 = std::min(tri.max_y, band * kBandHeight + kBandHeight - 1);
                    for (int y = y0; y <= y1; ++y) {
                        int64_t py = (int64_t(y) << kSubpixelBits) + kSubpixel / 2;
                        int64_t px = (int64_t(tri.min_x) << kSubpixelBits) + kSubpixel / 2;
                        int64_t w[3];
                        for (int e = 0; e < 3; ++e) w[e] = A[e] * px + B[e] * py + C[e];
                        for (int x = tri.min_x; x <= tri.max_x; ++x) {
                            if (((w[0] + bias[0]) | (w[1] + bias[1]) | (w[2] + bias[2])) >= 0) {
                                float l1 = float(w[1]) * inv_area, l2 = float(w[2]) * inv_area;
                                float l0 = 1.0f - l1 - l2;
                                float z = l0 * tri.z[0] + l1 * tri.z[1] + l2 * tri.z[2];
                                float& depth = fb.depth_at(x, y);
                                if (!state.depth_test || z < depth) {
                                    if (!write_color) {
                                        if (state.depth_write) depth = z;
                                    } else {
                                        float iw = l0 * tri.inv_w[0] + l1 * tri.inv_w[1] + l2 * tri.inv_w[2];
                                        V in = (tri.var[0] * l0 + tri.var[1] * l1 + tri.var[2] * l2) * (1.0f / iw);
                                        Vec3 color;
                                        if (shader.fragment(in, color)) {
                                            if (state.depth_write) depth = z;
                                            fb.pixel(x, y) = pack_rgba(color);
                                        }
                                    }
                                }
                            }
                            for (int e = 0; e < 3; ++e) w[e] += A[e] << kSubpixelBits;
                        }
                    }
                }
            }
        }
    });
    return stats;
}

}  // namespace sr
