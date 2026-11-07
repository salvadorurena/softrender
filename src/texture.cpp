#include "sr/texture.hpp"

#include <cmath>
#include <cstring>

namespace sr {

const char* texture_filter_name(TextureFilter f) {
    switch (f) {
        case TextureFilter::Nearest: return "nearest";
        case TextureFilter::Bilinear: return "bilinear";
        case TextureFilter::Trilinear: return "trilinear";
        case TextureFilter::Anisotropic: return "aniso";
        default: return "?";
    }
}

bool parse_texture_filter(const char* s, TextureFilter& out) {
    for (int i = 0; i < int(TextureFilter::Count); ++i)
        if (std::strcmp(s, texture_filter_name(TextureFilter(i))) == 0) {
            out = TextureFilter(i);
            return true;
        }
    return false;
}

namespace {
inline int wrap(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}
}  // namespace

Texture::Texture(int w, int h, Vec3 fill) {
    levels_.push_back({w, h, std::vector<Vec3>(size_t(w) * h, fill)});
    build_mipmaps();
}

void Texture::build_mipmaps() {
    levels_.resize(1);
    while (levels_.back().w > 1 || levels_.back().h > 1) {
        const Level& src = levels_.back();
        Level dst{std::max(1, src.w / 2), std::max(1, src.h / 2), {}};
        dst.texels.resize(size_t(dst.w) * dst.h);
        for (int y = 0; y < dst.h; ++y)
            for (int x = 0; x < dst.w; ++x) {
                // 2x2 box filter; clamps for odd sizes and 1-texel-wide levels.
                int x0 = std::min(2 * x, src.w - 1), x1 = std::min(2 * x + 1, src.w - 1);
                int y0 = std::min(2 * y, src.h - 1), y1 = std::min(2 * y + 1, src.h - 1);
                const Vec3* row0 = &src.texels[size_t(y0) * src.w];
                const Vec3* row1 = &src.texels[size_t(y1) * src.w];
                dst.texels[size_t(y) * dst.w + x] = (row0[x0] + row0[x1] + row1[x0] + row1[x1]) * 0.25f;
            }
        levels_.push_back(std::move(dst));
    }
}

Vec3 Texture::sample_nearest(Vec2 uv, int level) const {
    const Level& l = levels_[level];
    // v = 0 is the bottom of the image (OBJ convention), rows are stored top-down.
    int x = wrap(int(std::floor(uv.x * l.w)), l.w);
    int y = wrap(int(std::floor((1.0f - uv.y) * l.h)), l.h);
    return l.texels[size_t(y) * l.w + x];
}

Vec3 Texture::sample_bilinear(Vec2 uv, int level) const {
    const Level& l = levels_[level];
    float fx = uv.x * l.w - 0.5f;
    float fy = (1.0f - uv.y) * l.h - 0.5f;
    float x0f = std::floor(fx), y0f = std::floor(fy);
    float tx = fx - x0f, ty = fy - y0f;
    int x0 = wrap(int(x0f), l.w), x1 = wrap(int(x0f) + 1, l.w);
    int y0 = wrap(int(y0f), l.h), y1 = wrap(int(y0f) + 1, l.h);
    const Vec3* r0 = &l.texels[size_t(y0) * l.w];
    const Vec3* r1 = &l.texels[size_t(y1) * l.w];
    return lerp(lerp(r0[x0], r0[x1], tx), lerp(r1[x0], r1[x1], tx), ty);
}

Vec3 Texture::sample_trilinear(Vec2 uv, float lod) const {
    lod = std::clamp(lod, 0.0f, float(mip_count() - 1));
    int lo = int(lod);
    int hi = std::min(lo + 1, mip_count() - 1);
    float t = lod - float(lo);
    Vec3 a = sample_bilinear(uv, lo);
    return t > 0.0f ? lerp(a, sample_bilinear(uv, hi), t) : a;
}

float Texture::lod(Vec2 duv_dx, Vec2 duv_dy) const {
    // Footprint axes in level-0 texels; the longer one decides the level.
    float sx = std::hypot(duv_dx.x * width(), duv_dx.y * height());
    float sy = std::hypot(duv_dy.x * width(), duv_dy.y * height());
    float rho = std::max(sx, sy);
    return rho > 0.0f ? std::log2(rho) : 0.0f;
}

Vec3 Texture::sample(Vec2 uv, Vec2 duv_dx, Vec2 duv_dy, TextureFilter filter) const {
    switch (filter) {
        case TextureFilter::Nearest: return sample_nearest(uv);
        case TextureFilter::Bilinear: return sample_bilinear(uv);
        case TextureFilter::Trilinear: return sample_trilinear(uv, lod(duv_dx, duv_dy));
        default: break;
    }
    // Anisotropic: a pixel's footprint on a surface seen at a grazing angle is a
    // long thin ellipse. Pick the mip level from the *short* axis (keeping detail)
    // and cover the long axis with several trilinear taps along it.
    float sx = std::hypot(duv_dx.x * width(), duv_dx.y * height());
    float sy = std::hypot(duv_dy.x * width(), duv_dy.y * height());
    float major = std::max(sx, sy), minor = std::min(sx, sy);
    if (major <= 1e-8f) return sample_bilinear(uv);
    float ratio = minor > 1e-8f ? major / minor : float(kMaxAnisotropy);
    int taps = std::clamp(int(std::ceil(ratio)), 1, kMaxAnisotropy);
    float level = std::log2(major / float(taps));
    if (taps == 1) return sample_trilinear(uv, level);
    Vec2 axis = sx >= sy ? duv_dx : duv_dy;
    Vec3 sum(0.0f);
    for (int i = 0; i < taps; ++i) {
        float t = (i + 0.5f) / float(taps) - 0.5f;
        sum += sample_trilinear(uv + axis * t, level);
    }
    return sum * (1.0f / float(taps));
}

}  // namespace sr
