#pragma once

#include "viger_sr/image.hpp"

#include <filesystem>

namespace viger::sr {

// Portable PPM support is always available. Windows builds additionally use
// Windows Imaging Component for common PNG/JPEG/BMP/TIFF files.
RGB8 load_image(const std::filesystem::path& path);
void save_image(const RGB8& image, const std::filesystem::path& path);

} // namespace viger::sr
