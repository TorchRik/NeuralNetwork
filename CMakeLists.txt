cmake_minimum_required(VERSION 3.25)

project(NeuralNetwork)

add_library(MySources INTERFACE)
target_include_directories(MySources INTERFACE src)

add_library(NeuralNetwork INTERFACE)
target_include_directories(NeuralNetwork INTERFACE include)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

target_link_libraries(NeuralNetwork INTERFACE MySources)

add_executable(my_binary main.cpp)
target_link_libraries(my_binary PRIVATE NeuralNetwork)