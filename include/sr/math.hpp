// Small, header-only linear algebra for the renderer.
// Conventions: right-handed world space, column vectors (v' = M * v),
// OpenGL-style clip space (x, y, z in [-w, w]).
#pragma once

#include <algorithm>
#include <cmath>

namespace sr {

constexpr float kPi = 3.14159265358979323846f;

inline float radians(float deg) { return deg * (kPi / 180.0f); }
inline float clamp01(float x) { return std::clamp(x, 0.0f, 1.0f); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

struct Vec2 {
    float x = 0, y = 0;
    Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}
    Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
};

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    constexpr Vec3(float s) : x(s), y(s), z(s) {}
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(Vec3 o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return *this * (1.0f / s); }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(Vec3 o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};
inline Vec3 operator*(float s, Vec3 v) { return v * s; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(Vec3 v) {
    float len2 = dot(v, v);
    return len2 > 0 ? v * (1.0f / std::sqrt(len2)) : v;
}
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
inline Vec3 vmin(Vec3 a, Vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 vmax(Vec3 a, Vec3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
inline Vec3 reflect(Vec3 i, Vec3 n) { return i - n * (2.0f * dot(i, n)); }

struct Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
    Vec4() = default;
    constexpr Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr Vec4(Vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    Vec3 xyz() const { return {x, y, z}; }
    Vec4 operator+(Vec4 o) const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    Vec4 operator-(Vec4 o) const { return {x - o.x, y - o.y, z - o.z, w - o.w}; }
    Vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
};

// 4x4 matrix stored row-major: m[row][col].
struct Mat4 {
    float m[4][4] = {};

    static Mat4 identity() {
        Mat4 r;
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] +
                            m[i][2] * o.m[2][j] + m[i][3] * o.m[3][j];
        return r;
    }

    Vec4 operator*(Vec4 v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w};
    }

    Vec3 transform_point(Vec3 p) const { return (*this * Vec4(p, 1.0f)).xyz(); }
    // Correct for normals only when the upper 3x3 is a rotation + uniform scale.
    Vec3 transform_dir(Vec3 d) const { return (*this * Vec4(d, 0.0f)).xyz(); }

    static Mat4 translate(Vec3 t) {
        Mat4 r = identity();
        r.m[0][3] = t.x; r.m[1][3] = t.y; r.m[2][3] = t.z;
        return r;
    }
    static Mat4 scale(Vec3 s) {
        Mat4 r;
        r.m[0][0] = s.x; r.m[1][1] = s.y; r.m[2][2] = s.z; r.m[3][3] = 1.0f;
        return r;
    }
    static Mat4 rotate_y(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0][0] = c;  r.m[0][2] = s;
        r.m[2][0] = -s; r.m[2][2] = c;
        return r;
    }
    static Mat4 rotate_x(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[1][1] = c; r.m[1][2] = -s;
        r.m[2][1] = s; r.m[2][2] = c;
        return r;
    }

    static Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
        Vec3 f = normalize(target - eye);
        Vec3 s = normalize(cross(f, up));
        Vec3 u = cross(s, f);
        Mat4 r = identity();
        r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
        r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
        r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
        return r;
    }

    static Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
        float t = 1.0f / std::tan(fovy * 0.5f);
        Mat4 r;
        r.m[0][0] = t / aspect;
        r.m[1][1] = t;
        r.m[2][2] = (zfar + znear) / (znear - zfar);
        r.m[2][3] = (2.0f * zfar * znear) / (znear - zfar);
        r.m[3][2] = -1.0f;
        return r;
    }

    static Mat4 ortho(float l, float r_, float b, float t, float n, float f) {
        Mat4 r = identity();
        r.m[0][0] = 2.0f / (r_ - l);
        r.m[1][1] = 2.0f / (t - b);
        r.m[2][2] = -2.0f / (f - n);
        r.m[0][3] = -(r_ + l) / (r_ - l);
        r.m[1][3] = -(t + b) / (t - b);
        r.m[2][3] = -(f + n) / (f - n);
        return r;
    }
};

}  // namespace sr
