cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(heat)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

get_filename_component(OpenCL_INCLUDE "include" ABSOLUTE)
set(OpenCL_INCLUDE_DIR ${OpenCL_INCLUDE})

set(COMPUTECPP_USER_FLAGS -O3 -fsycl-split-modules=20 -mllvm -inline-threshold=10000 -no-serial-memop)


find_package(ComputeCpp REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


SET(SOURCES
        heat_sycl.cpp
)

SET(SOURCES2
        heat_sycl_1drange.cpp
)


add_executable(heat ${SOURCES})
add_executable(heat_1drange ${SOURCES2})

target_compile_options(heat
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
        -Wpedantic
        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes
        )

set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer -march=native -g)
set(RELEASE_OPTIONS -O3 -march=native)

target_compile_options(heat PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")


add_sycl_to_target(
        TARGET heat
        SOURCES ${SOURCES}
        ${CMAKE_CURRENT_BINARY_DIR}) # must be the last

add_sycl_to_target(
        TARGET heat_1drange
        SOURCES ${SOURCES2}
        ${CMAKE_CURRENT_BINARY_DIR}) # must be the last
