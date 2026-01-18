cmake_minimum_required(VERSION 3.18)
project(hpc_app LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hpc_app
    src/main.cpp
)

target_include_directories(hpc_app PRIVATE include)