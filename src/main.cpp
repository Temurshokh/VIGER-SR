#include "viger_sr/image_io.hpp"
#include "viger_sr/image_ops.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {
void print_help() {
    std::cout << "VIGER SR 0.2.0\n"
              << "Native lightweight super-resolution baseline\n\n"
              << "Usage:\n"
              << "  viger-sr <input> <output> [scale] [detail]\n\n"
              << "Examples:\n"
              << "  viger-sr photo.png enhanced.png 2 0.15\n"
              << "  viger-sr input.ppm output.ppm 4 0.10\n\n"
              << "On Windows, PNG/JPEG/BMP/TIFF input is decoded through WIC.\n";
}
}

int main(int argc, char** argv) {
    if (argc == 1) {
        print_help();
        return EXIT_SUCCESS;
    }

    if (argc < 3 || argc > 5) {
        print_help();
        return EXIT_FAILURE;
    }

    try {
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const int scale = argc >= 4 ? std::stoi(argv[3]) : 2;
        const float detail = argc >= 5 ? std::stof(argv[4]) : 0.15f;

        std::cout << "[VIGER SR] loading " << input_path << "...\n";
        const auto input = viger::sr::load_image(input_path);

        std::cout << "[VIGER SR] " << input.width() << 'x' << input.height()
                  << " -> " << input.width() * scale << 'x' << input.height() * scale << "\n";
        const auto output = viger::sr::tiny_sr_bicubic_baseline(input, scale, detail);

        viger::sr::save_image(output, output_path);
        std::cout << "[VIGER SR] wrote " << output_path << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[VIGER SR] error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
