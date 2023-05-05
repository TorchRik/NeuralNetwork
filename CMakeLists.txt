#cmake_minimum_required(VERSION 3.0)
#project(NeuralNetwork)
#
#set(EIGEN3_INCLUDE_DIR, /Users/torchrik/Documents/neural_network_from_zero/include)
#set(CMAKE_CXX_STANDARD 23)
#
## Найдите библиотеку Eigen и добавьте ее к проекту
#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})
#
## Добавьте ваш исходный код в проект
#add_executable(my_program main.cpp src/nonlinear_layer.h src/linear_layer.h src/activations_functions.h neural_network.h src/lose_functions.h src/layer.h)
#
## Свяжите вашу программу с библиотекой Eigen
#target_link_libraries(my_program Eigen3::Eigen)


######3

cmake_minimum_required(VERSION 3.5)

project(NeuralNetwork)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 21)

# Add the src directory as a subdirectory
add_subdirectory(src)

# Create a library called neural_network_lib using the source files in the src directory
add_library(NeuralNetwork neural_network.h src/activations_functions.h src/layer.h src/lose_functions.h)

