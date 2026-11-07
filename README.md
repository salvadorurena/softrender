# softrender

A multithreaded software rasterizer written from scratch in C++20. It uses no GPU and no graphics API: every pixel below comes from my own triangle pipeline running on the CPU.

![Orbiting torus knot with dynamic shadows](docs/orbit.gif)

The core library has **no third-party dependencies**. It includes its own math, OBJ loader, texture sampler and PNG encoder. SDL3 is used only by the optional interactive viewer, to put the finished framebuffer in a window.

![Hero render](docs/hero.png)

## Features

- **Programmable pipeline.** Vertex and fragment shaders are plain C++ structs passed as template parameters, so each shader gets its own fully inlined copy of the pipeline.
- **Exact fixed-point rasterization.** Edge functions run on 8-bit subpixel integer coordinates and use the **top-left fill rule**. Meshes are watertight: no cracks or double-shaded pixels on shared edges, and a test checks this.
- **Homogeneous clipping** against the near and far planes, plus a guard band in x/y. It uses Sutherland-Hodgman clipping and only clips against the planes a triangle actually crosses, chosen by outcodes.
- **Perspective-correct interpolation** of all varyings, plus backface culling and an early depth test.
- **Multithreaded on every core.** Vertex shading and triangle setup are split across threads, and the screen is cut into 8-row bands dealt round-robin to threads. Each pixel belongs to exactly one thread, so there are no locks or atomics.
- **Shadow mapping.** An orthographic depth pass from the sun with front-face culling, a slope-scaled bias and 16-tap PCF for soft edges.
- **Shading.** Blinn-Phong lighting with hemispheric ambient, bilinear texture filtering, ACES tone mapping, distance fog, and a sky gradient computed from each view ray's elevation.
- **Correct color handling.** Lighting runs in linear space, output is sRGB-encoded through a lookup table, and supersampling (`--ssaa N`) averages in linear space.
- **Debug views** for normals, depth and UVs, plus a depth-tested wireframe overlay.
- **OBJ loader** that handles polygons, negative indices, all `v/vt/vn` face forms and vertex deduplication. It generates smooth normals when the file has none.

| Normals | Depth | UVs |
|:-:|:-:|:-:|
| ![normals](docs/normals.png) | ![depth](docs/depth.png) | ![uv](docs/uv.png) |
| **Wireframe overlay** | **Shadows off** | |
| ![wireframe](docs/wireframe.png) | ![no shadows](docs/no_shadows.png) | |

## How a frame is drawn

```
vertex shader ─▶ primitive assembly ─▶ clip (near/far + guard band) ─▶ perspective divide
      ─▶ viewport + backface cull ─▶ fixed-point edge functions (top-left rule)
      ─▶ early depth test ─▶ perspective-correct varyings ─▶ fragment shader ─▶ sRGB framebuffer
```

Each frame runs three passes:
1. **Shadow pass:** depth only, rendered from the light into a 2048² shadow map.
2. **Main pass:** sky, then the ground and the model, drawn with the shader for the current view mode.
3. **Overlay (optional):** wireframe of the model's front faces, drawn as lines that are depth-tested against the main pass.

The whole pipeline lives in [`include/sr/rasterizer.hpp`](include/sr/rasterizer.hpp), about 270 lines. A shader looks like this:

```cpp
struct NormalShader {
    struct Varying {                       // anything that supports + and * float
        Vec3 normal;
        Varying operator+(const Varying& o) const { return {normal + o.normal}; }
        Varying operator*(float s) const { return {normal * s}; }
    };
    Mat4 mvp;
    Vec4 vertex(const Vertex& v, Varying& out) const {
        out.normal = v.normal;
        return mvp * Vec4(v.position, 1.0f);           // clip-space position
    }
    bool fragment(const Varying& in, Vec3& color) const {
        color = normalize(in.normal) * 0.5f + Vec3(0.5f);
        return true;                                    // false = discard
    }
};

sr::draw(framebuffer, NormalShader{mvp}, mesh.vertices, mesh.indices);
```

## Building

You need CMake 3.20 or newer and a C++20 compiler. SDL3 is optional and only needed for the viewer.

```bash
# macOS
brew install cmake sdl3
# Ubuntu / Debian (SDL3 is packaged on 25.04+; otherwise only the headless renderer builds)
sudo apt install cmake g++ libsdl3-dev

cmake -S . -B build
cmake --build build -j
ctest --test-dir build        # watertightness, culling, OBJ and PNG tests
```

## Usage

**Interactive viewer:**

```bash
./build/viewer                      # bundled torus knot
./build/viewer path/to/model.obj    # any OBJ: it is recentered, scaled and set on the ground
```

| Input | Action |
|---|---|
| Left drag / wheel | Orbit / zoom |
| Right drag | Move the sun |
| `1` `2` `3` `4` | Shaded / normals / depth / UV |
| `W` `S` `T` | Toggle wireframe / shadows / textures |
| `R` `L` | Toggle model spin / orbiting sun |
| `+` `-` | Change the resolution scale |
| `P` | Save `screenshot.png` |

The window title shows the FPS, render time and triangle count.

**Headless renderer** (renders straight to PNG):

```bash
./build/softrender --out render.png --size 1920x1080 --ssaa 2
./build/softrender --mode normals --wireframe --yaw 60 --pitch 30
./build/softrender --model bunny.obj --light-yaw 120
./build/softrender --bench 100 --size 1920x1080
```

Run `./build/softrender --help` for all options.

## Performance

These are timings from `softrender --bench 100`: the default scene (a 16,384-triangle knot on a ground plane) with shadows on and the model spinning. Measured on an Apple Silicon Mac with 10 cores in a Release build.

| Resolution | Frame time | FPS |
|---|---|---|
| 1280×720 | 8.5 ms | ~118 |
| 1920×1080 | 16.6 ms | ~60 |

The shadow pass takes about 0.9 ms of each frame.

## Project layout

```
include/sr/
  math.hpp          vectors, matrices, projection / look-at
  rasterizer.hpp    the pipeline: clipping, setup, fixed-point raster, threading
  framebuffer.hpp   color + depth targets, sRGB encode/decode, SSAA downsample
  scene.hpp         camera, render settings, multi-pass renderer
  mesh.hpp          vertex/mesh types, OBJ I/O, procedural geometry
  texture.hpp       bilinear / nearest sampling
  png.hpp           dependency-free PNG writer
src/                implementations (shaders live in scene.cpp)
apps/               viewer (SDL3) and softrender (headless CLI)
tools/mkassets.cpp  regenerates assets/torus_knot.obj
tests/              rasterizer and file-format tests
```

## Ideas for next steps

- Mipmapping with per-pixel UV derivatives computed from 2×2 pixel quads
- SIMD (NEON/AVX) edge evaluation on 4×4 pixel blocks, plus hierarchical tile rejection
- Normal mapping (tangent-space) and loading textures from files
- A clipping test for triangles that cross the near plane

## License

MIT
