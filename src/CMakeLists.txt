cmake_minimum_required(VERSION 3.10)


project(CNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()


add_executable(CNN ./main.cpp ./classifier.cpp ./stb_image.h ./channal.cpp ./convneurnet.cpp ./stb_impl.cpp)
