#pragma once

#include "viger_sr/image.hpp"

#include <filesystem>

namespace viger::sr {

class OnnxSRModel {
public:
    explicit OnnxSRModel(const std::filesystem::path& model_path);
    ~OnnxSRModel();

    OnnxSRModel(const OnnxSRModel&) = delete;
    OnnxSRModel& operator=(const OnnxSRModel&) = delete;

    OnnxSRModel(OnnxSRModel&&) noexcept;
    OnnxSRModel& operator=(OnnxSRModel&&) noexcept;

    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return model_path_; }

    // The exported model is a same-resolution refinement network. Upscaling is performed
    // by the SR pipeline before this method is called.
    [[nodiscard]] RGB8 enhance(const RGB8& upscaled) const;

private:
    struct Impl;
    std::filesystem::path model_path_;
    Impl* impl_ = nullptr;
};

} // namespace viger::sr
