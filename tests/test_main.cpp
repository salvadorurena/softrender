// Minimal self-contained tests (no framework). Run with `ctest` or ./build/sr_tests.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>

#include "sr/mesh.hpp"
#include "sr/png.hpp"
#include "sr/rasterizer.hpp"
#include "sr/texture.hpp"

using namespace sr;

static int failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++failures;                                               \
            std::printf("  FAIL %s:%d: %s\n    ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

// Shader whose varying is the NDC position, so each fragment knows its pixel
// and can count how many times it was shaded.
struct CoverageShader {
    struct Varying {
        float x, y;
        Varying operator+(const Varying& o) const { return {x + o.x, y + o.y}; }
        Varying operator*(float s) const { return {x * s, y * s}; }
    };
    int width, height;
    std::vector<std::atomic<int>>* hits;
    Vec4 vertex(const Vec2& p, Varying& out) const {
        out = {p.x, p.y};
        return {p.x, p.y, 0.0f, 1.0f};
    }
    bool fragment(const Varying& in, Vec3& color) const {
        int px = int(std::floor((in.x * 0.5f + 0.5f) * width));
        int py = int(std::floor((0.5f - in.y * 0.5f) * height));
        (*hits)[size_t(py) * width + px].fetch_add(1, std::memory_order_relaxed);
        color = Vec3(1.0f);
        return true;
    }
};

// A jittered grid of triangles exactly tiling the screen, with random winding.
// The top-left rule must shade every pixel exactly once: no cracks, no double hits.
static void test_watertight(int width, int height, int cells, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-0.2f, 0.2f);  // small enough that every cell stays convex
    std::vector<Vec2> verts;
    for (int j = 0; j <= cells; ++j)
        for (int i = 0; i <= cells; ++i) {
            float fx = float(i), fy = float(j);
            if (i > 0 && i < cells) fx += jitter(rng);
            if (j > 0 && j < cells) fy += jitter(rng);
            verts.push_back({fx / cells * 2.0f - 1.0f, fy / cells * 2.0f - 1.0f});
        }
    std::vector<uint32_t> idx;
    for (int j = 0; j < cells; ++j)
        for (int i = 0; i < cells; ++i) {
            uint32_t a = j * (cells + 1) + i, b = a + 1, c = a + cells + 1, d = c + 1;
            bool flip = rng() & 1, diag = rng() & 1;
            uint32_t t[6] = {a, b, diag ? d : c, diag ? a : b, d, c};
            if (flip) { std::swap(t[1], t[2]); std::swap(t[4], t[5]); }
            idx.insert(idx.end(), t, t + 6);
        }

    std::vector<std::atomic<int>> hits(size_t(width) * height);
    Framebuffer fb(width, height);
    CoverageShader shader{width, height, &hits};
    draw(fb, shader, std::span<const Vec2>(verts), std::span<const uint32_t>(idx),
         {.cull = CullMode::None, .depth_test = false, .depth_write = false});

    int holes = 0, doubles = 0;
    for (auto& h : hits) {
        holes += h == 0;
        doubles += h > 1;
    }
    CHECK(holes == 0 && doubles == 0, "%dx%d, %d cells: %d uncovered pixels, %d covered more than once",
          width, height, cells, holes, doubles);
}

static void test_backface_culling() {
    std::vector<Vec2> verts = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f}};
    std::vector<uint32_t> ccw = {0, 1, 2}, cw = {0, 2, 1};
    std::vector<std::atomic<int>> hits(64 * 64);
    Framebuffer fb(64, 64);
    CoverageShader shader{64, 64, &hits};
    auto front = draw(fb, shader, std::span<const Vec2>(verts), std::span<const uint32_t>(ccw));
    auto back = draw(fb, shader, std::span<const Vec2>(verts), std::span<const uint32_t>(cw));
    CHECK(front.triangles_rasterized == 1, "CCW triangle should be front-facing");
    CHECK(back.triangles_rasterized == 0, "CW triangle should be culled");
}

static void test_obj_loader() {
    std::string path = std::string(SR_TEST_TMP) + "/quad_test.obj";
    {
        std::ofstream f(path);
        f << "# quad with negative indices and v//vn faces\n"
             "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\n"
             "f -4//1 -3//1 -2//1 -1//1\n";
    }
    Mesh m = load_obj(path);
    CHECK(m.triangle_count() == 2, "quad should fan into 2 triangles, got %zu", m.triangle_count());
    CHECK(m.vertices.size() == 4, "shared corners should dedupe to 4 vertices, got %zu", m.vertices.size());
    CHECK(std::abs(m.vertices[0].normal.z - 1.0f) < 1e-6f, "normal should be +Z");

    Mesh knot = make_torus_knot(2, 3, 1.0f, 0.28f, 32, 8);
    save_obj(knot, path);
    Mesh back = load_obj(path);
    CHECK(back.triangle_count() == knot.triangle_count(), "OBJ round trip lost triangles");
}

static void test_png() {
    std::string path = std::string(SR_TEST_TMP) + "/test.png";
    Framebuffer fb(37, 11);  // odd size on purpose
    fb.clear(Vec3(0.5f));
    CHECK(write_png(fb, path), "write_png failed");
    std::ifstream f(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)), {});
    CHECK(data.size() > 8 && data.compare(1, 3, "PNG") == 0, "bad PNG signature");
    // Every PNG ends with the IEND chunk and its fixed CRC.
    CHECK(data.size() > 12 && data.substr(data.size() - 8) == std::string("IEND\xAE\x42\x60\x82", 8), "bad IEND chunk");
}

// Varying = screen-linear UV ramp. Under an orthographic (w = 1) projection the
// derivatives are exact constants, so every pixel must report the same ddx / ddy.
struct DerivativeShader {
    struct Varying {
        float u, v;
        Varying operator+(const Varying& o) const { return {u + o.u, v + o.v}; }
        Varying operator*(float s) const { return {u * s, v * s}; }
    };
    struct Result { std::atomic<int> count{0}, bad{0}; };
    Result* result;
    Vec4 vertex(const Vec2& p, Varying& out) const {
        out = {p.x * 3.0f, p.y * 5.0f};  // u grows to the right, v grows upward
        return {p.x, p.y, 0.0f, 1.0f};
    }
    bool fragment(const Varying&, const Derivatives<Varying>& d, Vec3& color) const {
        // One pixel step in NDC is 2/W horizontally and -2/H vertically (y is down on screen).
        const float W = 128, H = 96;
        bool ok = std::abs(d.ddx.u - 3.0f * 2.0f / W) < 1e-4f && std::abs(d.ddx.v) < 1e-4f &&
                  std::abs(d.ddy.v + 5.0f * 2.0f / H) < 1e-4f && std::abs(d.ddy.u) < 1e-4f;
        result->count++;
        if (!ok) result->bad++;
        color = Vec3(1.0f);
        return true;
    }
};

static void test_derivatives() {
    // A single small triangle: most quads along its edges contain helper lanes,
    // so this also checks that extrapolated helpers produce correct derivatives.
    std::vector<Vec2> verts = {{-0.6f, -0.7f}, {0.8f, -0.2f}, {-0.1f, 0.9f}};
    std::vector<uint32_t> idx = {0, 1, 2};
    Framebuffer fb(128, 96);
    DerivativeShader::Result result;
    draw(fb, DerivativeShader{&result}, std::span<const Vec2>(verts), std::span<const uint32_t>(idx));
    CHECK(result.count > 1000, "triangle should cover many pixels, got %d", result.count.load());
    CHECK(result.bad == 0, "%d of %d fragments had wrong derivatives", result.bad.load(), result.count.load());
}

static void test_mipmaps() {
    Texture t = Texture::generate(16, 8, [](float u, float) { return Vec3(u < 0.5f ? 1.0f : 0.0f); });
    CHECK(t.mip_count() == 5, "16x8 should have 5 levels (16x8 .. 1x1), got %d", t.mip_count());
    CHECK(t.level_width(4) == 1 && t.level_height(4) == 1, "last level should be 1x1");
    CHECK(std::abs(t.at(0, 0, 4).x - 0.5f) < 1e-6f, "1x1 level should be the average, got %f", t.at(0, 0, 4).x);

    // LOD = log2(footprint in level-0 texels); the longer axis wins.
    Texture sq(64, 64);
    CHECK(std::abs(sq.lod({1.0f / 64, 0}, {0, 1.0f / 64})) < 1e-5f, "1 texel per pixel -> lod 0");
    CHECK(std::abs(sq.lod({4.0f / 64, 0}, {0, 1.0f / 64}) - 2.0f) < 1e-5f, "4 texels per pixel -> lod 2");
    CHECK(std::abs(sq.lod({0, 0}, {0, 16.0f / 64}) - 4.0f) < 1e-5f, "16 texels per pixel -> lod 4");

    // A footprint covering the whole texture must return its average color, for every filter
    // except nearest/bilinear (which ignore the footprint).
    Vec3 tri = t.sample({0.3f, 0.5f}, {1.0f, 0}, {0, 1.0f}, TextureFilter::Trilinear);
    CHECK(std::abs(tri.x - 0.5f) < 1e-5f, "trilinear at the top level should be 0.5, got %f", tri.x);
}

int main() {
    std::printf("watertight rasterization\n");
    test_watertight(64, 64, 8, 1);
    test_watertight(320, 240, 37, 2);
    test_watertight(1280, 720, 101, 3);
    test_watertight(97, 53, 13, 4);
    std::printf("backface culling\n");
    test_backface_culling();
    std::printf("quad derivatives\n");
    test_derivatives();
    std::printf("mipmaps\n");
    test_mipmaps();
    std::printf("OBJ loader\n");
    test_obj_loader();
    std::printf("PNG writer\n");
    test_png();
    std::printf(failures ? "%d FAILED\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
