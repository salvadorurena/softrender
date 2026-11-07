// The heart of the renderer: a programmable, multithreaded triangle pipeline.
//
//   vertex shader -> primitive assembly -> homogeneous clipping -> perspective divide
//   -> viewport + backface culling -> fixed-point edge-function rasterization
//   -> 8x8 tile classification -> 2x2 quads -> early depth test -> perspective-correct interpolation + derivatives
//   -> fragment shader
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
// or, to receive screen-space derivatives (for texture LOD, edge detection, ...):
//
//       bool fragment(const Varying& in, const Derivatives<Varying>& d, Vec3& color) const;
//
// Colors are linear and get sRGB-encoded on write, unless the shader declares
// `static constexpr bool kOutputsDisplay = true;` (values already meant for display).
//
// Shaders are template parameters, so the whole pipeline is inlined per shader.
#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
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

// Screen-space derivatives of a fragment's interpolated inputs, one pixel to the
// right (ddx) and one pixel down (ddy). Shaders opt in with the 3-argument
// fragment(in, derivatives, color) overload.
template <class V>
struct Derivatives {
    V ddx, ddy;
};

struct RasterStats {
    size_t triangles_in = 0;
    size_t triangles_rasterized = 0;  // after clipping and culling
    // 8x8 tiles inside triangle bounding boxes, by coarse classification:
    size_t tiles_skipped = 0;  // entirely outside the triangle: no pixel work at all
    size_t tiles_full = 0;     // entirely inside: per-pixel edge tests skipped
    size_t tiles_partial = 0;  // straddle an edge: tested quad by quad
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
constexpr int kBandHeight = 8;      // rows per work band
constexpr int kTileSize = 8;        // coarse-rasterization tile (must equal kBandHeight)
static_assert(kTileSize == kBandHeight, "tiles must line up with thread bands");

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
    float inv_area;
    // Edge functions E(p) = A*px + B*py + C (fixed point), computed once in setup.
    int64_t A[3], B[3], C[3], bias[3];
    // Offsets from a tile's top-left pixel to the corner where each edge function
    // is largest / smallest over the tile's 8x8 pixel centers.
    int64_t max_off[3], min_off[3];
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

    // 2. Primitive assembly, clipping, projection, culling, setup and binning.
    struct BandBins {
        std::vector<uint32_t> start;  // num_bands + 1 offsets into items
        std::vector<uint32_t> items;  // indices into this thread's triangle list
    };
    const int num_bands = (H + kBandHeight - 1) / kBandHeight;
    std::vector<std::vector<ScreenTri<V>>> bins(workers);
    std::vector<BandBins> band_bins(workers);
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
            tri.inv_area = 1.0f / float(area);
            // Pixel bounding box (pixel centers sit at +0.5), clamped to the viewport.
            auto lo = [](int64_t a, int64_t b, int64_t c) { return std::min({a, b, c}); };
            auto hi = [](int64_t a, int64_t b, int64_t c) { return std::max({a, b, c}); };
            tri.min_x = std::max(0, int((lo(tri.x[0], tri.x[1], tri.x[2]) - kSubpixel / 2) >> kSubpixelBits));
            tri.min_y = std::max(0, int((lo(tri.y[0], tri.y[1], tri.y[2]) - kSubpixel / 2) >> kSubpixelBits));
            tri.max_x = std::min(W - 1, int((hi(tri.x[0], tri.x[1], tri.x[2]) - kSubpixel / 2) >> kSubpixelBits) + 1);
            tri.max_y = std::min(H - 1, int((hi(tri.y[0], tri.y[1], tri.y[2]) - kSubpixel / 2) >> kSubpixelBits) + 1);
            if (tri.min_x > tri.max_x || tri.min_y > tri.max_y) return;

            // Edge setup, done once here instead of in every thread that touches the triangle.
            // E is linear in x and y, so over a tile's pixel centers it peaks at one corner
            // and bottoms out at the opposite one, picked by the signs of A and B.
            const int64_t span = int64_t(kTileSize - 1) << kSubpixelBits;
            for (int e = 0; e < 3; ++e) {
                int ia = (e + 1) % 3, ib = (e + 2) % 3;  // edge opposite vertex e
                tri.A[e] = -(tri.y[ib] - tri.y[ia]);
                tri.B[e] = tri.x[ib] - tri.x[ia];
                tri.C[e] = -(tri.A[e] * tri.x[ia] + tri.B[e] * tri.y[ia]);
                tri.bias[e] = is_top_left(tri.x[ia], tri.y[ia], tri.x[ib], tri.y[ib]) ? 0 : -1;
                tri.max_off[e] = (std::max<int64_t>(tri.A[e], 0) + std::max<int64_t>(tri.B[e], 0)) * span;
                tri.min_off[e] = (std::min<int64_t>(tri.A[e], 0) + std::min<int64_t>(tri.B[e], 0)) * span;
            }
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

        // Bin this thread's triangles by the 8-row bands they overlap, as a compact
        // CSR list: band b's triangles are items[start[b] .. start[b+1]), in order.
        BandBins& bb = band_bins[t];
        bb.start.assign(size_t(num_bands) + 1, 0);
        for (const ScreenTri<V>& tri : out)
            for (int b = tri.min_y / kBandHeight; b <= tri.max_y / kBandHeight; ++b) ++bb.start[b + 1];
        for (int b = 0; b < num_bands; ++b) bb.start[b + 1] += bb.start[b];
        bb.items.resize(bb.start[num_bands]);
        std::vector<uint32_t> fill(bb.start.begin(), bb.start.end() - 1);
        for (uint32_t i = 0; i < out.size(); ++i)
            for (int b = out[i].min_y / kBandHeight; b <= out[i].max_y / kBandHeight; ++b) bb.items[fill[b]++] = i;
    });
    for (auto& b : bins) stats.triangles_rasterized += b.size();

    // 3. Rasterization: 8x8 tiles, then 2x2 pixel quads. The screen is split into
    //    8-row bands that threads claim one at a time from a shared counter, so busy
    //    bands (dense geometry, expensive shading) don't hold up the others. Each
    //    band is owned by exactly one thread and walks its triangle lists in
    //    submission order: no locks on pixels, and results are deterministic.
    //
    //    Each tile in a triangle's bounding box is first classified against the
    //    three edges at its corners: tiles fully outside are skipped outright, and
    //    tiles fully inside skip the per-pixel coverage tests.
    //
    //    Like a GPU, all four lanes of a quad are interpolated even when some fall
    //    outside the triangle ("helper" lanes, extrapolated). Differences between
    //    neighbouring lanes give each pixel the screen-space derivatives of its
    //    inputs, which is what texture LOD selection needs.
    constexpr bool kWantsDerivatives = requires(const Shader& s, const V& v, const Derivatives<V>& d, Vec3& c) {
        { s.fragment(v, d, c) } -> std::convertible_to<bool>;
    };
    // Shaders that write display-referred colors (debug views) opt out of sRGB encoding.
    constexpr bool kDisplayOutput = requires { requires Shader::kOutputsDisplay; };
    const bool write_color = fb.has_color();
    // Each thread counts into a local copy and stores it once at the end, so the
    // shared array is never written concurrently and needs no cache-line padding.
    struct TileCounts {
        size_t skipped = 0, full = 0, partial = 0;
    };
    std::vector<TileCounts> tile_counts(workers);

    // Rasterize the part of `tri` inside `band`.
    auto raster_band = [&](const ScreenTri<V>& tri, int band, TileCounts& tiles) {
        // Tiles are exactly one band tall, and bands start on even rows, so
        // neither tiles nor quads ever straddle two bands (or two threads).
        const int ty = band * kBandHeight;
        const int y0 = std::max(tri.min_y, ty) & ~1;
        const int y1 = std::min(tri.max_y, ty + kTileSize - 1);
        const int64_t tile_py = (int64_t(ty) << kSubpixelBits) + kSubpixel / 2;

        for (int tx = tri.min_x & ~(kTileSize - 1); tx <= tri.max_x; tx += kTileSize) {
            // Classify the tile against all three edges before touching any pixel.
            const int64_t tile_px = (int64_t(tx) << kSubpixelBits) + kSubpixel / 2;
            bool outside = false, full = true;
            for (int e = 0; e < 3; ++e) {
                int64_t corner = tri.A[e] * tile_px + tri.B[e] * tile_py + tri.C[e] + tri.bias[e];
                outside |= corner + tri.max_off[e] < 0;  // every pixel fails this edge
                full &= corner + tri.min_off[e] >= 0;    // every pixel passes this edge
            }
            if (outside) {
                ++tiles.skipped;
                continue;
            }
            ++(full ? tiles.full : tiles.partial);

            const int x0 = std::max(tri.min_x, tx) & ~1;
            const int x1 = std::min(tri.max_x, tx + kTileSize - 1);
            for (int y = y0; y <= y1; y += 2) {
                int64_t py = (int64_t(y) << kSubpixelBits) + kSubpixel / 2;
                int64_t px = (int64_t(x0) << kSubpixelBits) + kSubpixel / 2;
                int64_t w[3];  // edge values at the quad's top-left pixel
                for (int e = 0; e < 3; ++e) w[e] = tri.A[e] * px + tri.B[e] * py + tri.C[e];

                for (int x = x0; x <= x1; x += 2) {
                    // Lanes: 0 = (x, y), 1 = (x+1, y), 2 = (x, y+1), 3 = (x+1, y+1).
                    int64_t lw[4][3];
                    unsigned live = 0;
                    for (int lane = 0; lane < 4; ++lane) {
                        int dx = lane & 1, dy = lane >> 1;
                        for (int e = 0; e < 3; ++e)
                            lw[lane][e] = w[e] + (tri.A[e] << kSubpixelBits) * dx + (tri.B[e] << kSubpixelBits) * dy;
                        bool inside = full || ((lw[lane][0] + tri.bias[0]) | (lw[lane][1] + tri.bias[1]) |
                                               (lw[lane][2] + tri.bias[2])) >= 0;
                        if (inside && x + dx < fb.width() && y + dy < fb.height()) live |= 1u << lane;
                    }
                    for (int e = 0; e < 3; ++e) w[e] += tri.A[e] << (kSubpixelBits + 1);
                    if (!live) continue;

                    // Barycentrics and depth for all four lanes; early depth test on live ones.
                    float l[4][3], z[4];
                    for (int lane = 0; lane < 4; ++lane) {
                        l[lane][1] = float(lw[lane][1]) * tri.inv_area;
                        l[lane][2] = float(lw[lane][2]) * tri.inv_area;
                        l[lane][0] = 1.0f - l[lane][1] - l[lane][2];
                        z[lane] = l[lane][0] * tri.z[0] + l[lane][1] * tri.z[1] + l[lane][2] * tri.z[2];
                        if ((live >> lane & 1) && state.depth_test &&
                            !(z[lane] < fb.depth_at(x + (lane & 1), y + (lane >> 1))))
                            live &= ~(1u << lane);
                    }
                    if (!live) continue;

                    if (!write_color) {
                        if (state.depth_write)
                            for (int lane = 0; lane < 4; ++lane)
                                if (live >> lane & 1) fb.depth_at(x + (lane & 1), y + (lane >> 1)) = z[lane];
                        continue;
                    }

                    // Perspective-correct varyings. Helper lanes are only needed for derivatives.
                    V in[4];
                    for (int lane = 0; lane < 4; ++lane) {
                        if (!kWantsDerivatives && !(live >> lane & 1)) continue;
                        float iw = l[lane][0] * tri.inv_w[0] + l[lane][1] * tri.inv_w[1] +
                                   l[lane][2] * tri.inv_w[2];
                        iw = std::max(iw, 1e-8f);  // extrapolated helpers can approach w = 0
                        in[lane] = (tri.var[0] * l[lane][0] + tri.var[1] * l[lane][1] +
                                    tri.var[2] * l[lane][2]) * (1.0f / iw);
                    }

                    for (int lane = 0; lane < 4; ++lane) {
                        if (!(live >> lane & 1)) continue;
                        int dx = lane & 1, dy = lane >> 1;
                        Vec3 color;
                        bool keep;
                        if constexpr (kWantsDerivatives) {
                            // "Fine" derivatives: differences along this lane's own row / column.
                            Derivatives<V> d{in[dy * 2 + 1] + in[dy * 2] * -1.0f, in[2 + dx] + in[dx] * -1.0f};
                            keep = shader.fragment(in[lane], d, color);
                        } else {
                            keep = shader.fragment(in[lane], color);
                        }
                        if (keep) {
                            if (state.depth_write) fb.depth_at(x + dx, y + dy) = z[lane];
                            fb.pixel(x + dx, y + dy) = kDisplayOutput ? pack_rgba_display(color) : pack_rgba(color);
                        }
                    }
                }
            }
        }
    };

    std::atomic<int> next_band{0};
    parallel_for(workers, [&](unsigned t) {
        TileCounts tiles;
        for (int band; (band = next_band.fetch_add(1, std::memory_order_relaxed)) < num_bands;)
            for (unsigned src = 0; src < workers; ++src) {  // setup threads in submission order
                const BandBins& bb = band_bins[src];
                for (uint32_t k = bb.start[band]; k < bb.start[band + 1]; ++k)
                    raster_band(bins[src][bb.items[k]], band, tiles);
            }
        tile_counts[t] = tiles;
    });
    for (const TileCounts& c : tile_counts) {
        stats.tiles_skipped += c.skipped;
        stats.tiles_full += c.full;
        stats.tiles_partial += c.partial;
    }
    return stats;
}

}  // namespace sr
