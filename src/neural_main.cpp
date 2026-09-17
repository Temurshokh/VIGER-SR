#include "viger_sr/image_io.hpp"
#include "viger_sr/image_ops.hpp"
#include "viger_sr/onnx_sr.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {
void print_help() {
    std::cout << "VIGER SR Neural 0.3.0\n"
              << "Bicubic upscale + trained ONNX residual refinement\n\n"
              << "Usage:\n"
              << "  viger-sr-neural <input> <output> <model.onnx> [scale] [tile]\n\n"
              << "Examples:\n"
              << "  viger-sr-neural photo.jpg enhanced.png model.onnx 2\n"
              << "  viger-sr-neural photo.jpg enhanced.png model.onnx 2 768\n\n"
              << "tile=0 disables tiled inference. A positive tile size uses up to 64px overlap.\n";
}
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        print_help();
        return argc == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    try {
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const std::string model_path = argv[3];
        const int scale = argc >= 5 ? std::stoi(argv[4]) : 2;
        const int tile_size = argc >= 6 ? std::stoi(argv[5]) : 0;

        if (scale < 1 || scale > 8) {
            throw std::invalid_argument("scale must be between 1 and 8");
        }
        if (tile_size < 0) {
            throw std::invalid_argument("tile must be zero or positive");
        }

        std::cout << "[VIGER SR] loading " << input_path << "...\n";
        const auto input = viger::sr::load_image(input_path);
        std::cout << "[VIGER SR] bicubic upscale: " << input.width() << 'x' << input.height()
                  << " -> " << input.width() * scale << 'x' << input.height() * scale << "\n";

        const auto enlarged = viger::sr::upscale_bicubic(input, scale);
        viger::sr::OnnxSRModel model(model_path);
        if (!model.available()) {
            throw std::runtime_error("ONNX Runtime is unavailable in this build");
        }

        std::cout << "[VIGER SR] neural refinement: " << model.path().string() << "\n";
        const auto output = tile_size == 0
            ? model.enhance(enlarged)
            : model.enhance_tiled(enlarged, tile_size, std::min(64, std::max(0, tile_size / 3)));

        viger::sr::save_image(output, output_path);
        std::cout << "[VIGER SR] wrote " << output_path << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[VIGER SR] error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
