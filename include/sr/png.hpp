#pragma once

#include <string>

#include "sr/framebuffer.hpp"

namespace sr {

// Writes the framebuffer's color buffer as a 24-bit PNG. Self-contained encoder
// (no zlib): image data goes into uncompressed "stored" deflate blocks.
// Returns false if the file could not be written.
bool write_png(const Framebuffer& fb, const std::string& path);

}  // namespace sr
