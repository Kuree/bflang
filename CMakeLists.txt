cmake_minimum_required(VERSION 3.10)
project(bflang C CXX)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0116)
    # Introduced in cmake 3.20
    # https://cmake.org/cmake/help/latest/policy/CMP0116.html
    cmake_policy(SET CMP0116 OLD)
endif()

find_package(LLVM REQUIRED CONFIG)
set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
set(Clang_DIR ${LLVM_LIBRARY_DIR}/cmake/clang)
find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)
include(${Clang_DIR}/AddClang.cmake)

# include globally
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
