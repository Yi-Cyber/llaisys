cmake_minimum_required(VERSION 3.10)
project(llaisys)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找所有源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 输出位置设置在 lib 目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 生成动态库
add_library(llaisys SHARED ${SOURCES})
