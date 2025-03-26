cmake_minimum_required(VERSION 3.16)
project(Tensor CXX)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS_DEBUG "-g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -Ofast -DNDEBUG")
endif()

add_executable(tensor src/main.cpp src/Symmetry.cpp include/Tensor.hpp)

target_include_directories(tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


