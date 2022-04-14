cmake_minimum_required(VERSION 3.21)
project(h_style)

set(CMAKE_CXX_STANDARD 23)

add_executable(h_style main.cpp VectorOperations.h Tensor.h Tests.h TimeTests.h)
