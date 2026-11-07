// Color + depth render target. Colors are shaded in linear space and stored
// as 8-bit sRGB (RGBA byte order), ready to upload to a window or write to PNG.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "sr/math.hpp"

namespace sr {

namespace detail {
// Linear [0,1] quantised to 12 bits -> 8-bit sRGB. Avoids a pow() per channel per pixel.
inline std::array<uint8_t, 4096> make_linear_to_srgb() {
    std::array<uint8_t, 4096> t{};
    for (int i = 0; i < 4096; ++i) {
        float c = i / 4095.0f;
        float s = c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        t[i] = static_cast<uint8_t>(std::lround(clamp01(s) * 255.0f));
    }
    return t;
}
inline std::array<float, 256> make_srgb_to_linear() {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) {
        float s = i / 255.0f;
        t[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    }
    return t;
}
// Namespace-scope tables are built once at startup. (A function-local static
// would be lazily initialised, which costs a thread-safe guard check per lookup.)
inline const std::array<uint8_t, 4096> kLinearToSrgb = make_linear_to_srgb();
inline const std::array<float, 256> kSrgbToLinear = make_srgb_to_linear();
}  // namespace detail

inline uint8_t encode_srgb(float linear) {
    return detail::kLinearToSrgb[static_cast<int>(clamp01(linear) * 4095.0f + 0.5f)];
}
inline float decode_srgb(uint8_t s) { return detail::kSrgbToLinear[s]; }

inline uint32_t pack_rgba(Vec3 linear) {
    return uint32_t(encode_srgb(linear.x)) | uint32_t(encode_srgb(linear.y)) << 8 |
           uint32_t(encode_srgb(linear.z)) << 16 | 0xFF000000u;
}
// For values that are already display-referred (debug visualizations): plain
// 8-bit quantization, no transfer function.
inline uint32_t pack_rgba_display(Vec3 c) {
    auto q = [](float v) { return uint32_t(clamp01(v) * 255.0f + 0.5f); };
    return q(c.x) | q(c.y) << 8 | q(c.z) << 16 | 0xFF000000u;
}
inline Vec3 unpack_rgba(uint32_t p) {
    return {decode_srgb(p & 0xFF), decode_srgb((p >> 8) & 0xFF), decode_srgb((p >> 16) & 0xFF)};
}

class Framebuffer {
public:
    Framebuffer() = default;
    Framebuffer(int w, int h, bool with_color = true) { resize(w, h, with_color); }

    void resize(int w, int h, bool with_color = true) {
        width_ = w;
        height_ = h;
        depth_.assign(size_t(w) * h, 1.0f);
        color_.assign(with_color ? size_t(w) * h : 0, 0xFF000000u);
    }

    void clear(Vec3 linear_color, float depth = 1.0f) {
        std::fill(depth_.begin(), depth_.end(), depth);
        std::fill(color_.begin(), color_.end(), pack_rgba(linear_color));
    }
    void clear_depth(float depth = 1.0f) { std::fill(depth_.begin(), depth_.end(), depth); }

    // Vertical gradient background, top -> bottom.
    void clear_gradient(Vec3 top, Vec3 bottom) {
        clear_depth();
        for (int y = 0; y < height_; ++y) {
            uint32_t c = pack_rgba(lerp(top, bottom, float(y) / float(height_ - 1)));
            std::fill_n(color_.begin() + size_t(y) * width_, width_, c);
        }
    }

    int width() const { return width_; }
    int height() const { return height_; }
    bool has_color() const { return !color_.empty(); }

    float& depth_at(int x, int y) { return depth_[size_t(y) * width_ + x]; }
    float depth_at(int x, int y) const { return depth_[size_t(y) * width_ + x]; }
    uint32_t& pixel(int x, int y) { return color_[size_t(y) * width_ + x]; }
    uint32_t pixel(int x, int y) const { return color_[size_t(y) * width_ + x]; }

    const uint32_t* color_data() const { return color_.data(); }
    const float* depth_data() const { return depth_.data(); }

    // Box-filter downsample (used for supersampled anti-aliasing). Averages in linear space.
    Framebuffer downsample(int factor) const {
        Framebuffer out(width_ / factor, height_ / factor);
        const float inv = 1.0f / float(factor * factor);
        for (int y = 0; y < out.height(); ++y)
            for (int x = 0; x < out.width(); ++x) {
                Vec3 sum(0.0f);
                for (int sy = 0; sy < factor; ++sy)
                    for (int sx = 0; sx < factor; ++sx)
                        sum += unpack_rgba(pixel(x * factor + sx, y * factor + sy));
                out.pixel(x, y) = pack_rgba(sum * inv);
            }
        return out;
    }

private:
    int width_ = 0, height_ = 0;
    std::vector<uint32_t> color_;
    std::vector<float> depth_;
};

}  // namespace sr
