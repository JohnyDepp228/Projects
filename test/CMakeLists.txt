cmake_minimum_required(VERSION 3.14)
project(CNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT MSVC)
    add_link_options(-static)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()


include(FetchContent)

if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) 
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2 
)
FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(tests ./test.cpp)


target_sources(tests PRIVATE 
   
)

target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main)

add_test(NAME MyGTests COMMAND tests)

target_include_directories(tests PRIVATE ../src .)
