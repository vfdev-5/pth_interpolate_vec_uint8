cmake_minimum_required(VERSION 3.10)
project(check)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -mavx2 -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-address-use-after-scope -shared-libasan -g")

add_executable(check_mem_boundaries_with_asan check_mem_boundaries_with_asan.cpp)
set_property(TARGET check_mem_boundaries_with_asan PROPERTY CXX_STANDARD 17)