cmake_minimum_required(VERSION 3.20)
project(VIGER_SR VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VIGER_SR_BUILD_TESTS "Build tests" ON)

add_executable(viger-sr
    src/main.cpp
    src/image.cpp
    src/image_ops.cpp
    src/ppm.cpp
)

target_include_directories(viger-sr PRIVATE include)

target_compile_features(viger-sr PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(viger-sr PRIVATE /W4 /permissive-)
else()
    target_compile_options(viger-sr PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (VIGER_SR_BUILD_TESTS)
    enable_testing()
    add_executable(viger-sr-tests
        tests/test_image.cpp
        src/image.cpp
        src/image_ops.cpp
        src/ppm.cpp
    )
    target_include_directories(viger-sr-tests PRIVATE include)
    add_test(NAME viger-sr-tests COMMAND viger-sr-tests)
endif()
