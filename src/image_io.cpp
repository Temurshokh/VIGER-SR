#include "viger_sr/image_io.hpp"

#include "viger_sr/ppm.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace viger::sr {
namespace {

std::string extension_lower(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

#ifdef _WIN32
using Microsoft::WRL::ComPtr;

class ComInit {
public:
    ComInit() : result_(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComInit() {
        if (SUCCEEDED(result_)) {
            CoUninitialize();
        }
    }
    HRESULT result() const noexcept { return result_; }

private:
    HRESULT result_;
};

void check_hr(HRESULT result, const char* what) {
    if (FAILED(result)) {
        throw std::runtime_error(std::string(what) + " failed (HRESULT=" +
                                 std::to_string(static_cast<unsigned long>(result)) + ")");
    }
}

ComPtr<IWICImagingFactory> make_factory() {
    ComInit init;
    if (FAILED(init.result()) && init.result() != RPC_E_CHANGED_MODE) {
        check_hr(init.result(), "CoInitializeEx");
    }

    ComPtr<IWICImagingFactory> factory;
    check_hr(CoCreateInstance(
                 CLSID_WICImagingFactory,
                 nullptr,
                 CLSCTX_INPROC_SERVER,
                 IID_PPV_ARGS(&factory)),
             "CoCreateInstance(CLSID_WICImagingFactory)");
    return factory;
}

RGB8 load_wic(const std::filesystem::path& path) {
    ComInit init;
    if (FAILED(init.result()) && init.result() != RPC_E_CHANGED_MODE) {
        check_hr(init.result(), "CoInitializeEx");
    }

    ComPtr<IWICImagingFactory> factory;
    check_hr(CoCreateInstance(
                 CLSID_WICImagingFactory,
                 nullptr,
                 CLSCTX_INPROC_SERVER,
                 IID_PPV_ARGS(&factory)),
             "CoCreateInstance(CLSID_WICImagingFactory)");

    ComPtr<IWICBitmapDecoder> decoder;
    check_hr(factory->CreateDecoderFromFilename(
                 path.wstring().c_str(),
                 nullptr,
                 GENERIC_READ,
                 WICDecodeMetadataCacheOnLoad,
                 &decoder),
             "CreateDecoderFromFilename");

    ComPtr<IWICBitmapFrameDecode> frame;
    check_hr(decoder->GetFrame(0, &frame), "GetFrame");

    UINT width = 0;
    UINT height = 0;
    check_hr(frame->GetSize(&width, &height), "GetSize");

    ComPtr<IWICFormatConverter> converter;
    check_hr(factory->CreateFormatConverter(&converter), "CreateFormatConverter");
    check_hr(converter->Initialize(
                 frame.Get(),
                 GUID_WICPixelFormat24bppBGR,
                 WICBitmapDitherTypeNone,
                 nullptr,
                 0.0,
                 WICBitmapPaletteTypeMedianCut),
             "FormatConverter::Initialize");

    const UINT stride = width * 3;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(stride) * height);
    check_hr(converter->CopyPixels(
                 nullptr,
                 stride,
                 static_cast<UINT>(pixels.size()),
                 pixels.data()),
             "CopyPixels");

    RGB8 image(static_cast<int>(width), static_cast<int>(height), 3);
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * stride + x * 3;
            image.at(static_cast<int>(x), static_cast<int>(y), 0) = pixels[p + 2];
            image.at(static_cast<int>(x), static_cast<int>(y), 1) = pixels[p + 1];
            image.at(static_cast<int>(x), static_cast<int>(y), 2) = pixels[p + 0];
        }
    }
    return image;
}

void save_wic(const RGB8& image, const std::filesystem::path& path) {
    ComInit init;
    if (FAILED(init.result()) && init.result() != RPC_E_CHANGED_MODE) {
        check_hr(init.result(), "CoInitializeEx");
    }

    ComPtr<IWICImagingFactory> factory;
    check_hr(CoCreateInstance(
                 CLSID_WICImagingFactory,
                 nullptr,
                 CLSCTX_INPROC_SERVER,
                 IID_PPV_ARGS(&factory)),
             "CoCreateInstance(CLSID_WICImagingFactory)");

    const std::string ext = extension_lower(path);
    const GUID container = ext == ".jpg" || ext == ".jpeg"
                               ? GUID_ContainerFormatJpeg
                               : GUID_ContainerFormatPng;

    ComPtr<IWICStream> stream;
    check_hr(factory->CreateStream(&stream), "CreateStream");
    check_hr(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE),
             "InitializeFromFilename");

    ComPtr<IWICBitmapEncoder> encoder;
    check_hr(factory->CreateEncoder(container, nullptr, &encoder), "CreateEncoder");
    check_hr(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "Encoder::Initialize");

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> properties;
    check_hr(encoder->CreateNewFrame(&frame, &properties), "CreateNewFrame");
    check_hr(frame->Initialize(properties.Get()), "Frame::Initialize");
    check_hr(frame->SetSize(static_cast<UINT>(image.width()), static_cast<UINT>(image.height())),
             "Frame::SetSize");

    WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
    check_hr(frame->SetPixelFormat(&format), "SetPixelFormat");

    const UINT stride = static_cast<UINT>(image.width() * 3);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(stride) * image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * stride + x * 3;
            pixels[p + 0] = image.at(x, y, 2);
            pixels[p + 1] = image.at(x, y, 1);
            pixels[p + 2] = image.at(x, y, 0);
        }
    }

    check_hr(frame->WritePixels(
                 static_cast<UINT>(image.height()),
                 stride,
                 static_cast<UINT>(pixels.size()),
                 pixels.data()),
             "WritePixels");
    check_hr(frame->Commit(), "Frame::Commit");
    check_hr(encoder->Commit(), "Encoder::Commit");
}
#endif

} // namespace

RGB8 load_image(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Input image does not exist: " + path.string());
    }

    const std::string ext = extension_lower(path);
    if (ext == ".ppm" || ext == ".pnm") {
        return load_ppm(path.string());
    }

#ifdef _WIN32
    return load_wic(path);
#else
    throw std::runtime_error(
        "This build supports PPM/PNM only. PNG/JPEG/BMP/TIFF decoding is currently implemented for Windows.");
#endif
}

void save_image(const RGB8& image, const std::filesystem::path& path) {
    const std::string ext = extension_lower(path);
    if (ext == ".ppm" || ext == ".pnm") {
        save_ppm(image, path.string());
        return;
    }

#ifdef _WIN32
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        save_wic(image, path);
        return;
    }
#endif

    throw std::runtime_error(
        "Unsupported output format. Use .ppm, or .png/.jpg/.jpeg on Windows.");
}

} // namespace viger::sr
