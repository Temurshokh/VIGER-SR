#include "viger_sr/image_ops.hpp"
#include "viger_sr/onnx_sr.hpp"
#include "viger_sr/ppm.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {
void print_help() {
    std::cout << "VIGER SR Neural 0.2.0\n"
              << "Bicubic upscale + trained ONNX residual refinement\n\n"
              << "Usage:\n"
              << "  viger-sr-neural <input.ppm> <output.ppm> <model.onnx> [scale]\n";
}
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        print_help();
        return argc == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    try {
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const std::string model_path = argv[3];
        const int scale = argc == 5 ? std::stoi(argv[4]) : 2;

        std::cout << "[VIGER SR] loading " << input_path << "...\n";
        const auto input = viger::sr::load_ppm(input_path);
        std::cout << "[VIGER SR] bicubic upscale: " << input.width() << 'x' << input.height()
                  << " -> " << input.width() * scale << 'x' << input.height() * scale << "\n";

        const auto enlarged = viger::sr::upscale_bicubic(input, scale);
        viger::sr::OnnxSRModel model(model_path);
        if (!model.available()) {
            throw std::runtime_error("ONNX Runtime is unavailable in this build");
        }

        std::cout << "[VIGER SR] neural refinement: " << model.path().string() << "\n";
        const auto output = model.enhance(enlarged);
        viger::sr::save_ppm(output, output_path);
        std::cout << "[VIGER SR] wrote " << output_path << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[VIGER SR] error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
