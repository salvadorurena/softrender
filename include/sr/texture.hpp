#pragma once

#include <vector>

#include "sr/math.hpp"

namespace sr {

enum class TextureFilter { Nearest, Bilinear, Trilinear, Anisotropic, Count };
const char* texture_filter_name(TextureFilter f);
bool parse_texture_filter(const char* s, TextureFilter& out);

// RGB texture stored in linear space with a full mip chain, sampled with repeat wrapping.
class Texture {
public:
    static constexpr int kMaxAnisotropy = 8;

    Texture() = default;
    Texture(int w, int h, Vec3 fill = Vec3(1.0f));

    // Build a texture from a function f(u, v) -> linear RGB, evaluated at texel centers.
    template <class F>
    static Texture generate(int w, int h, F&& f) {
        Texture t(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) t.at(x, y) = f((x + 0.5f) / w, (y + 0.5f) / h);
        t.build_mipmaps();
        return t;
    }

    // Rebuilds levels 1..N from level 0 with a 2x2 box filter (in linear space).
    void build_mipmaps();

    // Filtered lookup. duv_dx / duv_dy are the UV derivatives across one pixel,
    // which decide the mip level (and, for anisotropic filtering, the footprint shape).
    Vec3 sample(Vec2 uv, Vec2 duv_dx, Vec2 duv_dy, TextureFilter filter) const;

    // Mip level of detail for a pixel footprint: log2 of its size in level-0 texels.
    float lod(Vec2 duv_dx, Vec2 duv_dy) const;

    Vec3 sample_nearest(Vec2 uv, int level = 0) const;
    Vec3 sample_bilinear(Vec2 uv, int level = 0) const;
    Vec3 sample_trilinear(Vec2 uv, float lod) const;

    int width() const { return levels_.empty() ? 0 : levels_[0].w; }
    int height() const { return levels_.empty() ? 0 : levels_[0].h; }
    int mip_count() const { return int(levels_.size()); }
    int level_width(int level) const { return levels_[level].w; }
    int level_height(int level) const { return levels_[level].h; }
    Vec3& at(int x, int y, int level = 0) { return levels_[level].texels[size_t(y) * levels_[level].w + x]; }
    const Vec3& at(int x, int y, int level = 0) const {
        return levels_[level].texels[size_t(y) * levels_[level].w + x];
    }

private:
    struct Level {
        int w = 0, h = 0;
        std::vector<Vec3> texels;
    };
    std::vector<Level> levels_;
};

}  // namespace sr
