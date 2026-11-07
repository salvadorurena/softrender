// Headless renderer: renders one frame to a PNG, or benchmarks the pipeline.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "sr/png.hpp"
#include "sr/scene.hpp"

using namespace sr;

static void usage() {
    std::puts(
        "usage: softrender [options]\n"
        "  --model PATH        OBJ file to render (default: bundled torus knot)\n"
        "  --out PATH          output PNG (default: render.png)\n"
        "  --size WxH          image size (default: 1280x720)\n"
        "  --ssaa N            supersample N x N and downsample (default: 1)\n"
        "  --mode MODE         shaded | normals | depth | uv (default: shaded)\n"
        "  --wireframe         overlay front-facing edges\n"
        "  --no-shadows        disable the shadow-map pass\n"
        "  --no-textures       untextured shading\n"
        "  --yaw DEG --pitch DEG --distance D   orbit camera\n"
        "  --light-yaw DEG --light-pitch DEG     sun direction\n"
        "  --spin DEG          rotate the model around Y\n"
        "  --bench N           render N frames and report timings instead of writing a PNG");
}

int main(int argc, char** argv) {
    std::string model_path, out_path = "render.png";
    int width = 1280, height = 720, ssaa = 1, bench = 0;
    OrbitCamera cam;
    RenderSettings rs;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--size") {
            if (std::sscanf(next(), "%dx%d", &width, &height) != 2 || width <= 0 || height <= 0) {
                std::fprintf(stderr, "bad --size, expected WxH\n");
                return 2;
            }
        } else if (a == "--ssaa") ssaa = std::max(1, std::atoi(next()));
        else if (a == "--mode") {
            if (!parse_view_mode(next(), rs.mode)) {
                std::fprintf(stderr, "unknown mode: %s\n", argv[i]);
                return 2;
            }
        } else if (a == "--wireframe") rs.wireframe = true;
        else if (a == "--no-shadows") rs.shadows = false;
        else if (a == "--no-textures") rs.textures = false;
        else if (a == "--yaw") cam.yaw = radians(std::strtof(next(), nullptr));
        else if (a == "--pitch") cam.pitch = radians(std::strtof(next(), nullptr));
        else if (a == "--distance") cam.distance = std::strtof(next(), nullptr);
        else if (a == "--light-yaw") rs.light_yaw = radians(std::strtof(next(), nullptr));
        else if (a == "--light-pitch") rs.light_pitch = radians(std::strtof(next(), nullptr));
        else if (a == "--spin") rs.model_spin = radians(std::strtof(next(), nullptr));
        else if (a == "--bench") bench = std::max(1, std::atoi(next()));
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    Scene scene;
    try {
        scene = Scene::create(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    cam.target = scene.focus;

    Renderer renderer;
    Framebuffer fb(width * ssaa, height * ssaa);

    if (bench) {
        renderer.render(fb, scene, cam, rs);  // warm-up
        double shadow = 0, main = 0, wire = 0;
        FrameStats s;
        auto t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < bench; ++f) {
            RenderSettings frame = rs;
            frame.model_spin += radians(360.0f) * f / bench;
            s = renderer.render(fb, scene, cam, frame);
            shadow += s.shadow_ms;
            main += s.main_ms;
            wire += s.wire_ms;
        }
        double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%dx%d, %s, %u threads, %zu model triangles, %d frames\n", fb.width(), fb.height(),
                    view_mode_name(rs.mode), worker_count(), s.model.triangles_in, bench);
        std::printf("  avg frame  %7.2f ms  (%.1f fps)\n", total / bench, 1000.0 * bench / total);
        std::printf("  shadow     %7.2f ms\n  main       %7.2f ms\n", shadow / bench, main / bench);
        if (rs.wireframe) std::printf("  wireframe  %7.2f ms\n", wire / bench);
        return 0;
    }

    FrameStats s = renderer.render(fb, scene, cam, rs);
    Framebuffer out = ssaa > 1 ? fb.downsample(ssaa) : std::move(fb);
    if (!write_png(out, out_path)) {
        std::fprintf(stderr, "error: could not write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d, %s, %.1f ms, %zu/%zu model tris rasterized)\n", out_path.c_str(), out.width(),
                out.height(), view_mode_name(rs.mode), s.total_ms(), s.model.triangles_rasterized, s.model.triangles_in);
    return 0;
}
