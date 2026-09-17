#include "viger_sr/onnx_sr.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef VIGER_SR_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace viger::sr {

struct OnnxSRModel::Impl {
#ifdef VIGER_SR_HAS_ONNX
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "viger-sr"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;

    explicit Impl(const std::filesystem::path& path) {
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(0);
#ifdef _WIN32
        session = std::make_unique<Ort::Session>(env, path.wstring().c_str(), options);
#else
        session = std::make_unique<Ort::Session>(env, path.string().c_str(), options);
#endif

        auto in_name = session->GetInputNameAllocated(0, allocator);
        auto out_name = session->GetOutputNameAllocated(0, allocator);
        if (!in_name || !out_name) {
            throw std::runtime_error("ONNX model has no input/output tensor");
        }
        input_name = in_name.get();
        output_name = out_name.get();
    }
#else
    explicit Impl(const std::filesystem::path&) {}
#endif
};

OnnxSRModel::OnnxSRModel(const std::filesystem::path& model_path)
    : model_path_(model_path) {
    if (!std::filesystem::exists(model_path_)) {
        throw std::runtime_error("Model file does not exist: " + model_path_.string());
    }
    impl_ = new Impl(model_path_);
}

OnnxSRModel::~OnnxSRModel() {
    delete impl_;
}

OnnxSRModel::OnnxSRModel(OnnxSRModel&& other) noexcept
    : model_path_(std::move(other.model_path_)), impl_(std::exchange(other.impl_, nullptr)) {}

OnnxSRModel& OnnxSRModel::operator=(OnnxSRModel&& other) noexcept {
    if (this != &other) {
        delete impl_;
        model_path_ = std::move(other.model_path_);
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}

bool OnnxSRModel::available() const noexcept {
#ifdef VIGER_SR_HAS_ONNX
    return impl_ != nullptr && impl_->session != nullptr;
#else
    return false;
#endif
}

RGB8 OnnxSRModel::enhance(const RGB8& upscaled) const {
    if (upscaled.empty()) {
        throw std::invalid_argument("Cannot run neural SR on an empty image");
    }
#ifdef VIGER_SR_HAS_ONNX
    if (!available()) {
        throw std::runtime_error("ONNX inference is not available in this build");
    }

    const std::int64_t height = upscaled.height();
    const std::int64_t width = upscaled.width();
    const std::array<std::int64_t, 4> shape{1, 3, height, width};
    const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    std::vector<float> input_data(plane * 3);
    for (int y = 0; y < upscaled.height(); ++y) {
        for (int x = 0; x < upscaled.width(); ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * upscaled.width() + x;
            input_data[p] = upscaled.at(x, y, 0) / 255.0f;
            input_data[plane + p] = upscaled.at(x, y, 1) / 255.0f;
            input_data[2 * plane + p] = upscaled.at(x, y, 2) / 255.0f;
        }
    }

    Ort::MemoryInfo memory_info("Cpu", OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());

    const char* input_names[] = {impl_->input_name.c_str()};
    const char* output_names[] = {impl_->output_name.c_str()};
    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("ONNX model returned no tensor output");
    }

    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();
    if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 3 ||
        output_shape[2] != height || output_shape[3] != width) {
        throw std::runtime_error("ONNX output shape does not match the input image");
    }

    const float* output_data = outputs[0].GetTensorData<float>();
    RGB8 result(upscaled.width(), upscaled.height(), 3);
    for (int y = 0; y < result.height(); ++y) {
        for (int x = 0; x < result.width(); ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * result.width() + x;
            for (int c = 0; c < 3; ++c) {
                const float value = std::clamp(
                    output_data[static_cast<std::size_t>(c) * plane + p], 0.0f, 1.0f);
                result.at(x, y, c) = static_cast<std::uint8_t>(value * 255.0f + 0.5f);
            }
        }
    }
    return result;
#else
    (void)upscaled;
    throw std::runtime_error(
        "This build has no ONNX Runtime support. Reconfigure with VIGER_SR_ENABLE_ONNX=ON.");
#endif
}

} // namespace viger::sr
