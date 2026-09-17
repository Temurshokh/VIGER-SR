#pragma once

#include "viger_sr/image.hpp"

#include <string>

namespace viger::sr {

RGB8 load_ppm(const std::string& path);
void save_ppm(const RGB8& image, const std::string& path);

} // namespace viger::sr
