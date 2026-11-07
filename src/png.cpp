#include "sr/png.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace sr {

namespace {

uint32_t crc32(const uint8_t* data, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    for (size_t i = 0; i < len; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

void put_u32be(std::vector<uint8_t>& out, uint32_t v) {
    out.insert(out.end(), {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)});
}

void write_chunk(FILE* f, const char type[4], const std::vector<uint8_t>& data) {
    std::vector<uint8_t> buf;
    buf.reserve(data.size() + 12);
    put_u32be(buf, uint32_t(data.size()));
    buf.insert(buf.end(), type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    uint32_t crc = crc32(buf.data() + 4, buf.size() - 4) ^ 0xFFFFFFFFu;
    put_u32be(buf, crc);
    std::fwrite(buf.data(), 1, buf.size(), f);
}

}  // namespace

bool write_png(const Framebuffer& fb, const std::string& path) {
    const int w = fb.width(), h = fb.height();

    // Raw scanlines: filter byte 0 (None) + RGB triples.
    std::vector<uint8_t> raw;
    raw.reserve(size_t(h) * (1 + 3 * size_t(w)));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        for (int x = 0; x < w; ++x) {
            uint32_t p = fb.pixel(x, y);
            raw.insert(raw.end(), {uint8_t(p), uint8_t(p >> 8), uint8_t(p >> 16)});
        }
    }

    // zlib stream: header, stored deflate blocks (<= 65535 bytes each), Adler-32.
    std::vector<uint8_t> z = {0x78, 0x01};
    for (size_t pos = 0; pos < raw.size() || raw.empty();) {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + n == raw.size();
        z.insert(z.end(), {uint8_t(last), uint8_t(n), uint8_t(n >> 8), uint8_t(~n), uint8_t(~n >> 8)});
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
        if (last) break;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    put_u32be(z, (b << 16) | a);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::fwrite(signature, 1, 8, f);

    std::vector<uint8_t> ihdr;
    put_u32be(ihdr, uint32_t(w));
    put_u32be(ihdr, uint32_t(h));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit, truecolor RGB, deflate, no filter, no interlace
    write_chunk(f, "IHDR", ihdr);
    write_chunk(f, "IDAT", z);
    write_chunk(f, "IEND", {});
    bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

}  // namespace sr
