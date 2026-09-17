cmake_minimum_required(VERSION 3.20)
project(VIGER_SR VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VIGER_SR_BUILD_TESTS "Build tests" ON)
option(VIGER_SR_BUILD_CHAT "Build VIGER Chat foundation" ON)
option(VIGER_SR_ENABLE_ONNX "Enable ONNX Runtime neural inference" OFF)

add_library(viger_sr_core
    src/image_ops.cpp
    src/ppm.cpp
)

target_include_directories(viger_sr_core PUBLIC include)
target_compile_features(viger_sr_core PUBLIC cxx_std_17)

if (MSVC)
    target_compile_options(viger_sr_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(viger_sr_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(viger-sr src/main.cpp)
target_link_libraries(viger-sr PRIVATE viger_sr_core)

if (VIGER_SR_ENABLE_ONNX)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation root")
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib")

    if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or disable VIGER_SR_ENABLE_ONNX.")
    endif()

    target_sources(viger_sr_core PRIVATE src/onnx_sr.cpp)
    target_include_directories(viger_sr_core PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(viger_sr_core PRIVATE "${ONNXRUNTIME_LIBRARY}")
    target_compile_definitions(viger_sr_core PRIVATE VIGER_SR_HAS_ONNX=1)

    add_executable(viger-sr-neural src/neural_main.cpp)
    target_link_libraries(viger-sr-neural PRIVATE viger_sr_core)
endif()

if (VIGER_SR_BUILD_CHAT)
    add_executable(viger-chat
        src/chat_main.cpp
        src/chat.cpp
    )
    target_include_directories(viger-chat PRIVATE include)
    target_compile_features(viger-chat PRIVATE cxx_std_17)
    if (MSVC)
        target_compile_options(viger-chat PRIVATE /W4 /permissive-)
    else()
        target_compile_options(viger-chat PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if (VIGER_SR_BUILD_TESTS)
    enable_testing()
    add_executable(viger-sr-tests
        tests/test_image.cpp
        src/image_ops.cpp
        src/ppm.cpp
    )
    target_include_directories(viger-sr-tests PRIVATE include)
    add_test(NAME viger-sr-tests COMMAND viger-sr-tests)

    if (VIGER_SR_BUILD_CHAT)
        add_executable(viger-chat-tests
            tests/test_chat.cpp
            src/chat.cpp
        )
        target_include_directories(viger-chat-tests PRIVATE include)
        add_test(NAME viger-chat-tests COMMAND viger-chat-tests)
    endif()
endif()
