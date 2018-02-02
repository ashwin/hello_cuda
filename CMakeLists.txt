cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11")
 
cuda_add_executable(
    hello_cuda 
    src/main.cu
)
