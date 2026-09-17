#include "viger_sr/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace viger::sr {
namespace {

std::uint8_t clamp_u8(float value) {
    return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

float sample(const RGB8& image, float x, float y, int c) {
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, image.width() - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, image.height() - 1);
    const int x1 = std::clamp(x0 + 1, 0, image.width() - 1);
    const int y1 = std::clamp(y0 + 1, 0, image.height() - 1);

    const float fx = std::clamp(x - std::floor(x), 0.0f, 1.0f);
    const float fy = std::clamp(y - std::floor(y), 0.0f, 1.0f);

    const float a = static_cast<float>(image.at(x0, y0, c));
    const float b = static_cast<float>(image.at(x1, y0, c));
    const float d = static_cast<float>(image.at(x0, y1, c));
    const float e = static_cast<float>(image.at(x1, y1, c));

    return (a * (1.0f - fx) + b * fx) * (1.0f - fy)
         + (d * (1.0f - fx) + e * fx) * fy;
}

float luminance(const RGB8& image, int x, int y) {
    return 0.2126f * image.at(x, y, 0)
         + 0.7152f * image.at(x, y, 1)
         + 0.0722f * image.at(x, y, 2);
}

} // namespace

RGB8 upscale_bilinear(const RGB8& input, int scale) {
    if (input.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (input.channels() != 3) {
        throw std::invalid_argument("RGB8 image must have 3 channels");
    }
    if (scale < 1 || scale > 8) {
        throw std::invalid_argument("Scale must be between 1 and 8");
    }

    RGB8 output(input.width() * scale, input.height() * scale, 3);

    for (int y = 0; y < output.height(); ++y) {
        const float source_y = static_cast<float>(y) / scale;
        for (int x = 0; x < output.width(); ++x) {
            const float source_x = static_cast<float>(x) / scale;
            for (int c = 0; c < 3; ++c) {
                output.at(x, y, c) = clamp_u8(sample(input, source_x, source_y, c));
            }
        }
    }

    return output;
}

RGB8 edge_aware_sharpen(const RGB8& input, float amount) {
    if (input.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    amount = std::clamp(amount, 0.0f, 2.0f);

    RGB8 output(input.width(), input.height(), 3);

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(input.width() - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(input.height() - 1, y + 1);

            const float center_luma = luminance(input, x, y);
            const float avg_luma = (
                luminance(input, xm, y) + luminance(input, xp, y) +
                luminance(input, x, ym) + luminance(input, x, yp)) * 0.25f;
            const float edge = center_luma - avg_luma;

            for (int c = 0; c < 3; ++c) {
                output.at(x, y, c) = clamp_u8(
                    static_cast<float>(input.at(x, y, c)) + amount * edge);
            }
        }
    }

    return output;
}

RGB8 tiny_sr_baseline(const RGB8& input, int scale, float detail_amount) {
    auto enlarged = upscale_bilinear(input, scale);
    return edge_aware_sharpen(enlarged, detail_amount);
}

} // namespace viger::sr
