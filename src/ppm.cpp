#include "viger_sr/image.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace viger::sr {

namespace {
void skip_ws(std::istream& in) {
    while (std::isspace(in.peek())) in.get();
    if (in.peek() == '#') {
        std::string line;
        std::getline(in, line);
        skip_ws(in);
    }
}
}

RGB8 load_ppm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input: " + path);

    std::string magic;
    in >> magic;
    if (magic != "P6") throw std::runtime_error("Only binary PPM (P6) is supported");

    skip_ws(in);
    int width = 0, height = 0, max_value = 0;
    in >> width;
    skip_ws(in);
    in >> height;
    skip_ws(in);
    in >> max_value;
    in.get(); // consume one whitespace byte after max value

    if (width <= 0 || height <= 0 || max_value != 255) {
        throw std::runtime_error("Unsupported PPM dimensions or max value");
    }

    RGB8 image(width, height, 3);
    in.read(reinterpret_cast<char*>(image.data().data()), static_cast<std::streamsize>(image.data().size()));
    if (in.gcount() != static_cast<std::streamsize>(image.data().size())) {
        throw std::runtime_error("Unexpected end of PPM file");
    }
    return image;
}

void save_ppm(const RGB8& image, const std::string& path) {
    if (image.channels() != 3) throw std::invalid_argument("PPM output requires RGB image");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + path);

    out << "P6\n" << image.width() << ' ' << image.height() << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.data().data()), static_cast<std::streamsize>(image.data().size()));
    if (!out) throw std::runtime_error("Failed to write output image");
}

} // namespace viger::sr
