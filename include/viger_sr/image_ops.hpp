#pragma once

#include "viger_sr/image.hpp"

namespace viger::sr {

RGB8 upscale_bilinear(const RGB8& input, int scale);
RGB8 edge_aware_sharpen(const RGB8& input, float amount);
RGB8 tiny_sr_baseline(const RGB8& input, int scale, float detail_amount = 0.20f);

} // namespace viger::sr
