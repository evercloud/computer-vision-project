cmake_minimum_required(VERSION 3.10)
project(document_scanner)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ask CMake to find an installed OpenCV package.
# On Ubuntu: sudo apt install libopencv-dev
# On macOS with Homebrew: brew install opencv
find_package(OpenCV REQUIRED)

# Inform CMake where OpenCV headers live.
include_directories(${OpenCV_INCLUDE_DIRS})

# Collect all source files.
set(SOURCES
    main.cpp
    scanner.cpp
    utils.cpp
)

add_executable(document_scanner ${SOURCES})

# Link against the OpenCV runtime libraries.
target_link_libraries(document_scanner ${OpenCV_LIBS})
