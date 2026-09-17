#include "viger_sr/image.hpp"
#include "viger_sr/image_ops.hpp"

#include <cassert>

int main() {
    viger::sr::RGB8 image(2, 2, 3, 0);
    image.at(0, 0, 0) = 255;
    image.at(1, 0, 1) = 255;
    image.at(0, 1, 2) = 255;
    image.at(1, 1, 0) = 255;

    const auto upscaled = viger::sr::upscale_bilinear(image, 2);
    assert(upscaled.width() == 4);
    assert(upscaled.height() == 4);
    assert(upscaled.channels() == 3);

    const auto enhanced = viger::sr::tiny_sr_baseline(image, 2);
    assert(enhanced.width() == 4);
    assert(enhanced.height() == 4);

    return 0;
}
