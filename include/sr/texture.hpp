#pragma once

#include <vector>

#include "sr/math.hpp"

namespace sr {

// RGB texture stored in linear space, sampled with repeat wrapping.
class Texture {
public:
    Texture() = default;
    Texture(int w, int h, Vec3 fill = Vec3(1.0f)) : width_(w), height_(h), texels_(size_t(w) * h, fill) {}

    // Build a texture from a function f(u, v) -> linear RGB, evaluated at texel centers.
    template <class F>
    static Texture generate(int w, int h, F&& f) {
        Texture t(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) t.at(x, y) = f((x + 0.5f) / w, (y + 0.5f) / h);
        return t;
    }

    Vec3 sample_nearest(Vec2 uv) const;
    Vec3 sample_bilinear(Vec2 uv) const;

    int width() const { return width_; }
    int height() const { return height_; }
    Vec3& at(int x, int y) { return texels_[size_t(y) * width_ + x]; }
    const Vec3& at(int x, int y) const { return texels_[size_t(y) * width_ + x]; }

private:
    int width_ = 0, height_ = 0;
    std::vector<Vec3> texels_;
};

}  // namespace sr
