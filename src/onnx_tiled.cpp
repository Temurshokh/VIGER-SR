#include "viger_sr/onnx_sr.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace viger::sr {
namespace {

RGB8 crop(const RGB8& image, int x0, int y0, int width, int height) {
    RGB8 tile(width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                tile.at(x, y, c) = image.at(x0 + x, y0 + y, c);
            }
        }
    }
    return tile;
}

float feather_weight(int coordinate, int length, int overlap) {
    if (overlap <= 0 || length <= 1) return 1.0f;
    const int distance_to_edge = std::min(coordinate, length - 1 - coordinate);
    if (distance_to_edge >= overlap) return 1.0f;

    const float normalized = std::clamp(
        static_cast<float>(distance_to_edge + 1) / static_cast<float>(overlap + 1),
        0.0f,
        1.0f);
    return 0.05f + 0.95f * (0.5f - 0.5f * std::cos(normalized * 3.14159265358979323846f));
}

} // namespace

RGB8 OnnxSRModel::enhance_tiled(const RGB8& upscaled, int tile_size, int overlap) const {
    if (upscaled.empty()) throw std::invalid_argument("Cannot tile an empty image");
    if (tile_size <= 0) throw std::invalid_argument("tile_size must be positive");
    if (overlap < 0 || overlap * 2 >= tile_size) {
        throw std::invalid_argument("overlap must be >= 0 and less than half tile_size");
    }

    if (upscaled.width() <= tile_size && upscaled.height() <= tile_size) {
        return enhance(upscaled);
    }

    const int width = upscaled.width();
    const int height = upscaled.height();
    const int step = tile_size - overlap;

    std::vector<float> accum(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, 0.0f);
    std::vector<float> weights(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0.0f);

    for (int y0 = 0;;) {
        const int tile_y = std::min(y0, std::max(0, height - tile_size));
        const int tile_h = std::min(tile_size, height - tile_y);

        for (int x0 = 0;;) {
            const int tile_x = std::min(x0, std::max(0, width - tile_size));
            const int tile_w = std::min(tile_size, width - tile_x);
            const RGB8 tile = crop(upscaled, tile_x, tile_y, tile_w, tile_h);
            const RGB8 enhanced = enhance(tile);

            for (int y = 0; y < tile_h; ++y) {
                for (int x = 0; x < tile_w; ++x) {
                    const float wx = feather_weight(x, tile_w, std::min(overlap, tile_w / 2));
                    const float wy = feather_weight(y, tile_h, std::min(overlap, tile_h / 2));
                    const float weight = wx * wy;
                    const int out_x = tile_x + x;
                    const int out_y = tile_y + y;
                    const std::size_t pixel =
                        static_cast<std::size_t>(out_y) * static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(out_x);
                    weights[pixel] += weight;
                    const std::size_t base = pixel * 3u;
                    for (int c = 0; c < 3; ++c) {
                        accum[base + static_cast<std::size_t>(c)] +=
                            static_cast<float>(enhanced.at(x, y, c)) * weight;
                    }
                }
            }

            if (tile_x + tile_w >= width) break;
            x0 += step;
        }

        if (tile_y + tile_h >= height) break;
        y0 += step;
    }

    RGB8 result(width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const float weight = std::max(weights[pixel], 1e-6f);
            const std::size_t base = pixel * 3u;
            for (int c = 0; c < 3; ++c) {
                result.at(x, y, c) = static_cast<std::uint8_t>(std::clamp(
                    accum[base + static_cast<std::size_t>(c)] / weight, 0.0f, 255.0f));
            }
        }
    }
    return result;
}

} // namespace viger::sr
