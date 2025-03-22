cmake_minimum_required(VERSION 3.16)
project(Tensor CXX)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GCC")
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS_DEBUG "-g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -Ofast -DNDEBUG")
endif()

add_executable(codegen src/codegen/main.cpp)

set(GENERATED_HEADER ${CMAKE_BINARY_DIR}/include/tensor.hpp)

add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    COMMAND codegen ${GENERATED_HEADER}
    DEPENDS codegen
    COMMENT "Generating header file"
    VERBATIM
)
add_custom_target(generate_header
    DEPENDS ${GENERATED_HEADER}
)
add_executable(test src/test/main.cpp ${GENERATED_HEADER})
add_dependencies(test generate_header)
target_include_directories(test PRIVATE ${CMAKE_BINARY_DIR}/include)

