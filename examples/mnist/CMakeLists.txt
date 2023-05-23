cmake_minimum_required(VERSION 3.14)
project(NeuralNetworkMnist LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(NeuralNetwork REQUIRED)
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(NeuralNetworkMnist src/mnist_parser.cpp)
target_sources(NeuralNetworkMnist PRIVATE ${sources})
target_link_libraries(NeuralNetworkMnist PRIVATE NeuralNetwork::NeuralNetwork)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(NeuralNetworkMnist NeuralNetwork::NeuralNetwork)
endif()
