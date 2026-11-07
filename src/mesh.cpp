#include "sr/mesh.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sr {

void Mesh::bounds(Vec3& lo, Vec3& hi) const {
    lo = Vec3(1e30f);
    hi = Vec3(-1e30f);
    for (const Vertex& v : vertices) {
        lo = vmin(lo, v.position);
        hi = vmax(hi, v.position);
    }
}

void Mesh::normalize_to(float size) {
    if (vertices.empty()) return;
    Vec3 lo, hi;
    bounds(lo, hi);
    Vec3 center = (lo + hi) * 0.5f, ext = hi - lo;
    float largest = std::max({ext.x, ext.y, ext.z});
    float s = largest > 0 ? size / largest : 1.0f;
    for (Vertex& v : vertices) v.position = (v.position - center) * s;
}

void Mesh::compute_normals() {
    for (Vertex& v : vertices) v.normal = Vec3(0.0f);
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        Vertex& a = vertices[indices[i]];
        Vertex& b = vertices[indices[i + 1]];
        Vertex& c = vertices[indices[i + 2]];
        // Unnormalized cross product = area weighting for free.
        Vec3 n = cross(b.position - a.position, c.position - a.position);
        a.normal += n;
        b.normal += n;
        c.normal += n;
    }
    for (Vertex& v : vertices) v.normal = normalize(v.normal);
}

void Mesh::orient_to_normals() {
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vertex& a = vertices[indices[i]];
        const Vertex& b = vertices[indices[i + 1]];
        const Vertex& c = vertices[indices[i + 2]];
        Vec3 face = cross(b.position - a.position, c.position - a.position);
        if (dot(face, a.normal + b.normal + c.normal) < 0) std::swap(indices[i + 1], indices[i + 2]);
    }
}

// ---------------------------------------------------------------------------
// OBJ

namespace {

struct IndexKey {
    int v, vt, vn;
    bool operator==(const IndexKey&) const = default;
};
struct IndexKeyHash {
    size_t operator()(const IndexKey& k) const {
        return (size_t(uint32_t(k.v)) * 73856093u) ^ (size_t(uint32_t(k.vt)) * 19349663u) ^
               (size_t(uint32_t(k.vn)) * 83492791u);
    }
};

// OBJ indices are 1-based; negative values count back from the end. Returns 0-based or -1.
int resolve_index(const std::string& s, size_t count) {
    if (s.empty()) return -1;
    int i = std::stoi(s);
    if (i > 0) return i - 1;
    if (i < 0) return int(count) + i;
    return -1;
}

}  // namespace

Mesh load_obj(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open OBJ file: " + path);

    std::vector<Vec3> positions, normals;
    std::vector<Vec2> uvs;
    Mesh mesh;
    std::unordered_map<IndexKey, uint32_t, IndexKeyHash> cache;
    bool has_normals = true;

    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        std::istringstream in(line);
        std::string tag;
        if (!(in >> tag) || tag[0] == '#') continue;

        if (tag == "v") {
            Vec3 p;
            in >> p.x >> p.y >> p.z;
            positions.push_back(p);
        } else if (tag == "vt") {
            Vec2 t;
            in >> t.x >> t.y;
            uvs.push_back(t);
        } else if (tag == "vn") {
            Vec3 n;
            in >> n.x >> n.y >> n.z;
            normals.push_back(normalize(n));
        } else if (tag == "f") {
            std::vector<uint32_t> face;
            std::string tok;
            while (in >> tok) {
                // Split "v/vt/vn" keeping empty fields ("v//vn").
                std::array<std::string, 3> parts;
                size_t field = 0, start = 0;
                for (size_t i = 0; i <= tok.size() && field < 3; ++i)
                    if (i == tok.size() || tok[i] == '/') {
                        parts[field++] = tok.substr(start, i - start);
                        start = i + 1;
                    }
                IndexKey key{resolve_index(parts[0], positions.size()), resolve_index(parts[1], uvs.size()),
                             resolve_index(parts[2], normals.size())};
                if (key.v < 0 || key.v >= int(positions.size()) || key.vt >= int(uvs.size()) ||
                    key.vn >= int(normals.size()))
                    throw std::runtime_error(path + ":" + std::to_string(line_no) + ": bad face index");
                if (key.vn < 0) has_normals = false;

                auto [it, inserted] = cache.try_emplace(key, uint32_t(mesh.vertices.size()));
                if (inserted) {
                    Vertex v;
                    v.position = positions[key.v];
                    if (key.vt >= 0) v.uv = uvs[key.vt];
                    if (key.vn >= 0) v.normal = normals[key.vn];
                    mesh.vertices.push_back(v);
                }
                face.push_back(it->second);
            }
            for (size_t k = 1; k + 1 < face.size(); ++k) {
                mesh.indices.push_back(face[0]);
                mesh.indices.push_back(face[k]);
                mesh.indices.push_back(face[k + 1]);
            }
        }
        // Other statements (o, g, s, usemtl, mtllib, ...) are ignored.
    }

    if (mesh.indices.empty()) throw std::runtime_error("OBJ file has no faces: " + path);
    if (!has_normals) mesh.compute_normals();
    return mesh;
}

void save_obj(const Mesh& mesh, const std::string& path, const std::string& comment) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write OBJ file: " + path);
    if (!comment.empty()) std::fprintf(f, "# %s\n", comment.c_str());
    for (const Vertex& v : mesh.vertices) std::fprintf(f, "v %.5f %.5f %.5f\n", v.position.x, v.position.y, v.position.z);
    for (const Vertex& v : mesh.vertices) std::fprintf(f, "vt %.5f %.5f\n", v.uv.x, v.uv.y);
    for (const Vertex& v : mesh.vertices) std::fprintf(f, "vn %.4f %.4f %.4f\n", v.normal.x, v.normal.y, v.normal.z);
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        uint32_t a = mesh.indices[i] + 1, b = mesh.indices[i + 1] + 1, c = mesh.indices[i + 2] + 1;
        std::fprintf(f, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", a, a, a, b, b, b, c, c, c);
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Procedural geometry

namespace {
// Builds a (cols+1) x (rows+1) grid of vertices and two triangles per cell.
// The extra seam column/row lets UVs wrap cleanly.
template <class F>
Mesh make_grid(int cols, int rows, F&& vertex_at) {
    Mesh m;
    m.vertices.reserve(size_t(cols + 1) * (rows + 1));
    for (int j = 0; j <= rows; ++j)
        for (int i = 0; i <= cols; ++i) m.vertices.push_back(vertex_at(float(i) / cols, float(j) / rows));
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < cols; ++i) {
            uint32_t a = j * (cols + 1) + i, b = a + 1, c = a + (cols + 1), d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    m.orient_to_normals();
    return m;
}
}  // namespace

Mesh make_plane(float size, float uv_repeat) {
    return make_grid(1, 1, [&](float u, float v) {
        return Vertex{{(u - 0.5f) * size, 0.0f, (v - 0.5f) * size}, {0, 1, 0}, {u * uv_repeat, v * uv_repeat}};
    });
}

Mesh make_uv_sphere(float radius, int rings, int segments) {
    return make_grid(segments, rings, [&](float u, float v) {
        float phi = u * 2.0f * kPi, theta = v * kPi;
        Vec3 n{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
        return Vertex{n * radius, n, {u, 1.0f - v}};
    });
}

Mesh make_torus_knot(int p, int q, float radius, float tube, int segments, int sides) {
    auto curve = [&](float t) {
        float r = 2.0f + std::cos(float(q) / float(p) * t);
        return Vec3{r * std::cos(t), r * std::sin(t), std::sin(float(q) / float(p) * t)} * (radius * 0.5f);
    };
    const float length = float(p) * 2.0f * kPi;
    Mesh m = make_grid(segments, sides, [&](float u, float v) {
        float t = u * length;
        Vec3 c0 = curve(t), c1 = curve(t + 0.01f);
        // Moving frame along the curve.
        Vec3 T = normalize(c1 - c0);
        Vec3 N = normalize(c1 + c0);
        Vec3 B = normalize(cross(T, N));
        N = cross(B, T);
        float a = v * 2.0f * kPi;
        Vec3 offset = N * (-std::cos(a)) + B * std::sin(a);
        return Vertex{c0 + offset * tube, normalize(offset), {u * float(segments) / float(sides) * 0.5f, v}};
    });
    return m;
}

}  // namespace sr
