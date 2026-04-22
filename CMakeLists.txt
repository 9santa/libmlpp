cmake_minimum_required(VERSION 3.16)

project(libmlpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(libmlpp INTERFACE)
target_include_directories(libmlpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_decision_tree tests/test_decision_tree.cpp)
target_link_libraries(test_decision_tree PRIVATE libmlpp)

add_executable(test_linear_classifier tests/test_linear_classifier.cpp)
target_link_libraries(test_linear_classifier PRIVATE libmlpp)
