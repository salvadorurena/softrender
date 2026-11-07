#include "sr/texture.hpp"

#include <cmath>

namespace sr {

namespace {
inline int wrap(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}
}  // namespace

Vec3 Texture::sample_nearest(Vec2 uv) const {
    // v = 0 is the bottom of the image (OBJ convention), rows are stored top-down.
    int x = wrap(int(std::floor(uv.x * width_)), width_);
    int y = wrap(int(std::floor((1.0f - uv.y) * height_)), height_);
    return at(x, y);
}

Vec3 Texture::sample_bilinear(Vec2 uv) const {
    float fx = uv.x * width_ - 0.5f;
    float fy = (1.0f - uv.y) * height_ - 0.5f;
    float x0f = std::floor(fx), y0f = std::floor(fy);
    float tx = fx - x0f, ty = fy - y0f;
    int x0 = wrap(int(x0f), width_), x1 = wrap(int(x0f) + 1, width_);
    int y0 = wrap(int(y0f), height_), y1 = wrap(int(y0f) + 1, height_);
    Vec3 top = lerp(at(x0, y0), at(x1, y0), tx);
    Vec3 bottom = lerp(at(x0, y1), at(x1, y1), tx);
    return lerp(top, bottom, ty);
}

}  // namespace sr
