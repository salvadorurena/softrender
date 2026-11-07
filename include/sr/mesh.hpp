#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sr/math.hpp"

namespace sr {

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;  // triangle list, counter-clockwise front faces

    size_t triangle_count() const { return indices.size() / 3; }
    void bounds(Vec3& lo, Vec3& hi) const;

    // Recenters on the origin and uniformly scales so the largest extent equals `size`.
    void normalize_to(float size);
    // Area-weighted smooth normals from the triangle geometry.
    void compute_normals();
    // Flips any triangle whose winding disagrees with its vertex normals.
    void orient_to_normals();
};

// Wavefront OBJ: v / vt / vn / f, polygons (fan-triangulated), negative indices,
// and the v, v/vt, v//vn, v/vt/vn face forms. Missing normals are generated.
// Throws std::runtime_error on failure.
Mesh load_obj(const std::string& path);
void save_obj(const Mesh& mesh, const std::string& path, const std::string& comment = "");

// Procedural geometry.
Mesh make_plane(float size, float uv_repeat);
Mesh make_torus_knot(int p, int q, float radius, float tube, int segments, int sides);
Mesh make_uv_sphere(float radius, int rings, int segments);

}  // namespace sr
