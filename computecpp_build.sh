#!/bin/bash

# This script will call the necessary CMake envokation on the Zoo for using ComputeCpp

module load gcc/9.1.0
module load cuda/9.2
module load cmake/3.14.5
module load computecpp/1.1.3

mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DComputeCpp_DIR=/nfs/software/x86_64/computecpp/1.1.3/ -DCOMPUTECPP_BITCODE=ptx64

