// Interactive viewer. SDL3 is used only to open a window and blit the finished
// framebuffer; every pixel is produced by the software pipeline.
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cstdio>
#include <exception>
#include <string>

#include "sr/png.hpp"
#include "sr/scene.hpp"

using namespace sr;

static const char* kHelp =
    "controls:\n"
    "  left drag    orbit camera        wheel    zoom\n"
    "  right drag   move the sun        1-5      shaded / normals / depth / uv / mip levels\n"
    "  W wireframe  S shadows  T textures  F texture filter  R spin model  L orbit sun\n"
    "  +/-          resolution scale    P        save screenshot.png\n"
    "  Esc          quit";

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "";
    Scene scene;
    try {
        scene = Scene::create(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("softrender", 1280, 720, SDL_WINDOW_RESIZABLE);
    SDL_Renderer* sdl = window ? SDL_CreateRenderer(window, nullptr) : nullptr;
    if (!sdl) {
        std::fprintf(stderr, "window creation failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_SetRenderVSync(sdl, 1);
    std::puts(kHelp);

    OrbitCamera cam;
    cam.target = scene.focus;
    RenderSettings rs;
    Renderer renderer;
    Framebuffer fb;
    SDL_Texture* texture = nullptr;
    float res_scale = 1.0f;
    bool spin = true, orbit_light = false, running = true;

    Uint64 last = SDL_GetPerformanceCounter(), title_timer = last;
    double frame_ms_accum = 0, render_ms_accum = 0;
    int frames = 0;

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_EVENT_QUIT: running = false; break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (e.motion.state & SDL_BUTTON_LMASK) {
                        cam.yaw -= e.motion.xrel * 0.008f;
                        cam.pitch = std::clamp(cam.pitch + e.motion.yrel * 0.008f, radians(-10.0f), radians(85.0f));
                    } else if (e.motion.state & SDL_BUTTON_RMASK) {
                        rs.light_yaw -= e.motion.xrel * 0.01f;
                        rs.light_pitch = std::clamp(rs.light_pitch - e.motion.yrel * 0.01f, radians(10.0f), radians(85.0f));
                    }
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    cam.distance = std::clamp(cam.distance * std::pow(0.9f, e.wheel.y), 1.5f, 30.0f);
                    break;
                case SDL_EVENT_KEY_DOWN:
                    switch (e.key.key) {
                        case SDLK_ESCAPE: running = false; break;
                        case SDLK_1: rs.mode = ViewMode::Shaded; break;
                        case SDLK_2: rs.mode = ViewMode::Normals; break;
                        case SDLK_3: rs.mode = ViewMode::Depth; break;
                        case SDLK_4: rs.mode = ViewMode::UV; break;
                        case SDLK_5: rs.mode = ViewMode::Mips; break;
                        case SDLK_F:
                            rs.filter = TextureFilter((int(rs.filter) + 1) % int(TextureFilter::Count));
                            break;
                        case SDLK_W: rs.wireframe = !rs.wireframe; break;
                        case SDLK_S: rs.shadows = !rs.shadows; break;
                        case SDLK_T: rs.textures = !rs.textures; break;
                        case SDLK_R: spin = !spin; break;
                        case SDLK_L: orbit_light = !orbit_light; break;
                        case SDLK_EQUALS:
                        case SDLK_KP_PLUS: res_scale = std::min(2.0f, res_scale + 0.25f); break;
                        case SDLK_MINUS:
                        case SDLK_KP_MINUS: res_scale = std::max(0.25f, res_scale - 0.25f); break;
                        case SDLK_P:
                            if (write_png(fb, "screenshot.png")) std::puts("saved screenshot.png");
                            break;
                        default: break;
                    }
                    break;
                default: break;
            }
        }

        Uint64 now = SDL_GetPerformanceCounter();
        float dt = float(double(now - last) / double(SDL_GetPerformanceFrequency()));
        last = now;
        if (spin) rs.model_spin += dt * 0.5f;
        if (orbit_light) rs.light_yaw += dt * 0.8f;

        // (Re)allocate the framebuffer to match the window and resolution scale.
        int ww, wh;
        SDL_GetWindowSize(window, &ww, &wh);
        int fw = std::max(1, int(ww * res_scale)), fh = std::max(1, int(wh * res_scale));
        if (!texture || fb.width() != fw || fb.height() != fh) {
            if (texture) SDL_DestroyTexture(texture);
            fb.resize(fw, fh);
            texture = SDL_CreateTexture(sdl, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, fw, fh);
            SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_NEAREST);
        }

        FrameStats stats = renderer.render(fb, scene, cam, rs);
        SDL_UpdateTexture(texture, nullptr, fb.color_data(), fb.width() * 4);
        SDL_RenderClear(sdl);
        SDL_RenderTexture(sdl, texture, nullptr, nullptr);
        SDL_RenderPresent(sdl);

        frame_ms_accum += dt * 1000.0;
        render_ms_accum += stats.total_ms();
        if (++frames, double(now - title_timer) / double(SDL_GetPerformanceFrequency()) > 0.5) {
            char title[256];
            std::snprintf(title, sizeof title,
                          "softrender | %s%s%s | %s | %dx%d | %.0f fps | render %.1f ms | %zu tris",
                          view_mode_name(rs.mode), rs.wireframe ? " + wire" : "",
                          rs.shadows && rs.mode == ViewMode::Shaded ? " + shadows" : "",
                          texture_filter_name(rs.filter), fw, fh,
                          1000.0 * frames / frame_ms_accum, render_ms_accum / frames,
                          stats.model.triangles_in + stats.ground.triangles_in);
            SDL_SetWindowTitle(window, title);
            frames = 0;
            frame_ms_accum = render_ms_accum = 0;
            title_timer = now;
        }
    }

    if (texture) SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(sdl);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
