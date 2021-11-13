cmake_minimum_required(VERSION 3.20)
project(untitled6)

set(CMAKE_CXX_STANDARD 14)

add_executable(untitled6 main.cpp)

find_package(MPI REQUIRED)
target_link_libraries(untitled6 PUBLIC MPI::MPI_CXX)