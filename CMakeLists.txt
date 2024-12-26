cmake_minimum_required(VERSION 3.10)
project(bfc)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED)
set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
find_package(MLIR REQUIRED)

add_subdirectory(include)
add_subdirectory(lib)
