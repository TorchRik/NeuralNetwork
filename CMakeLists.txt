cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

project(NeuralNetwork
        VERSION 0.0.1
        DESCRIPTION "Simple Neural Network implementation"
        LANGUAGES CXX)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------
include(cmake/utils.cmake)
include(GNUInstallDirs)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

# NEURAL_NETWORK_SHARED_LIBS option (undefined by default) can be used to force shared/static build
option(NEURAL_NETWORK_TESTS "Build NeuralNetwork tests" OFF)
option(NEURAL_NETWORK_BUILD_EXAMPLES "Build NeuralNetwork examples" OFF)
option(NEURAL_NETWORK_BUILD_DOCS "Build NeuralNetwork documentation" OFF)
option(NEURAL_NETWORK_INSTALL "Generate target for installing NeuralNetwork" ${is_top_level})
set_if_undefined(NEURAL_NETWORK_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetwork" CACHE STRING
        "Install path for NeuralNetwork package-related CMake files")

if (DEFINED NEURAL_NETWORK_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${NEURAL_NETWORK_SHARED_LIBS})
endif ()

if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(NeuralNetwork include/NeuralNetwork/neural_network.cpp)
add_library(NeuralNetwork::NeuralNetwork ALIAS NeuralNetwork)

#----------------------------------------------------------------------------------------------------------------------
# NeuralNetwork dependencies
#----------------------------------------------------------------------------------------------------------------------

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(NeuralNetwork Eigen3::Eigen)

include_directories(./EigenRand)


#----------------------------------------------------------------------------------------------------------------------
# NeuralNetwork sources
#----------------------------------------------------------------------------------------------------------------------

include(GenerateExportHeader)
set(export_file_name "export_shared.h")

if (NOT BUILD_SHARED_LIBS)
    set(export_file_name "export_static.h")
endif ()

generate_export_header(NeuralNetwork EXPORT_FILE_NAME include/NeuralNetwork/${export_file_name})

set(sources
        include/NeuralNetwork/
        include/NeuralNetwork/neural_network.h
        src/activations_functions.h
        src/layer.h
        src/layer.cpp
        src/lose_functions.h
        )
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# NeuralNetwork target
#----------------------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

target_sources(NeuralNetwork PRIVATE ${sources})
target_compile_definitions(NeuralNetwork PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:NEURAL_NETWORK_STATIC_DEFINE>")

target_include_directories(NeuralNetwork
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

set_target_properties(NeuralNetwork PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------------------------------------------------
# other targets
#----------------------------------------------------------------------------------------------------------------------

if (NEURAL_NETWORK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (NEURAL_NETWORK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (NEURAL_NETWORK_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs include)
endif ()
