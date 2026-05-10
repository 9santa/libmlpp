cmake_minimum_required(VERSION 3.16)

project(libmlpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(libmlpp INTERFACE)
target_include_directories(libmlpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(train_lenet_mnist examples/train_lenet_mnist.cpp)
target_link_libraries(train_lenet_mnist PRIVATE libmlpp)
target_compile_definitions(train_lenet_mnist PRIVATE
    MLPP_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

enable_testing()

function(add_mlpp_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE libmlpp)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_mlpp_test(test_decision_tree tests/test_decision_tree.cpp)
add_mlpp_test(test_linear_classifier tests/test_linear_classifier.cpp)
add_mlpp_test(test_svm tests/test_svm.cpp)
add_mlpp_test(test_lenet tests/test_lenet.cpp)
