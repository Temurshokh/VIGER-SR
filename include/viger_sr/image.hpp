#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viger::sr {

template <typename T>
class Image {
public:
    Image() = default;

    Image(int width, int height, int channels, T value = T{})
        : width_(width), height_(height), channels_(channels),
          data_(static_cast<std::size_t>(width) * height * channels, value) {
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw std::invalid_argument("Image dimensions/channels must be positive");
        }
    }

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    int channels() const noexcept { return channels_; }

    T& at(int x, int y, int c) {
        return data_.at(index(x, y, c));
    }

    const T& at(int x, int y, int c) const {
        return data_.at(index(x, y, c));
    }

    const std::vector<T>& data() const noexcept { return data_; }
    std::vector<T>& data() noexcept { return data_; }

    bool empty() const noexcept { return data_.empty(); }

private:
    std::size_t index(int x, int y, int c) const {
        if (x < 0 || x >= width_ || y < 0 || y >= height_ || c < 0 || c >= channels_) {
            throw std::out_of_range("Image coordinate out of range");
        }
        return (static_cast<std::size_t>(y) * width_ + x) * channels_ + c;
    }

    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    std::vector<T> data_;
};

using Gray8 = Image<std::uint8_t>;
using RGB8 = Image<std::uint8_t>;

} // namespace viger::sr
