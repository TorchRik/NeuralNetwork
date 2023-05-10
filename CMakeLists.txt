cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

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

# MYLIB_SHARED_LIBS option (undefined by default) can be used to force shared/static build
option(MYLIB_BUILD_TESTS "Build NeuralNetwork tests" OFF)
#option(MYLIB_BUILD_EXAMPLES "Build NeuralNetwork examples" OFF)
#option(MYLIB_BUILD_DOCS "Build NeuralNetwork documentation" OFF)
option(MYLIB_INSTALL "Generate target for installing NeuralNetwork" ${is_top_level})
set_if_undefined(MYLIB_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetwork" CACHE STRING
        "Install path for NeuralNetwork package-related CMake files")

if (DEFINED MYLIB_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${MYLIB_SHARED_LIBS})
endif ()

if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(NeuralNetwork include/NeuralNetwork/neural_network.cpp include/NeuralNetwork/saver.h src/constants.h) # initialized below
add_library(NeuralNetwork::NeuralNetwork ALIAS NeuralNetwork)

#----------------------------------------------------------------------------------------------------------------------
# NeuralNetwork dependencies
#----------------------------------------------------------------------------------------------------------------------

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(NeuralNetwork Eigen3::Eigen)

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
        src/lose_functions.h
        )
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# NeuralNetwork target
#----------------------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

target_sources(NeuralNetwork PRIVATE ${sources})
target_compile_definitions(NeuralNetwork PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:MYLIB_STATIC_DEFINE>")

target_include_directories(NeuralNetwork
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

set_target_properties(NeuralNetwork PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

#if (MYLIB_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
#    configure_package_config_file(cmake/mylib-config.cmake.in mylib-config.cmake
#            INSTALL_DESTINATION "${MYLIB_INSTALL_CMAKEDIR}")
#
#    write_basic_package_version_file(mylib-config-version.cmake
#            COMPATIBILITY SameMajorVersion)
#
#    install(TARGETS mylib EXPORT mylib_export
#            RUNTIME COMPONENT mylib
#            LIBRARY COMPONENT mylib NAMELINK_COMPONENT mylib-dev
#            ARCHIVE COMPONENT mylib-dev
#            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
#    install(DIRECTORY include/
#            TYPE INCLUDE
#            COMPONENT mylib-dev)
#    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/mylib/${export_file_name}"
#            COMPONENT mylib-dev
#            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mylib")
#
#    set(targets_file "mylib-shared-targets.cmake")
#
#    if (NOT BUILD_SHARED_LIBS)
#        set(targets_file "mylib-static-targets.cmake")
#    endif ()
#
#    install(EXPORT mylib_export
#            COMPONENT mylib-dev
#            FILE "${targets_file}"
#            DESTINATION "${MYLIB_INSTALL_CMAKEDIR}"
#            NAMESPACE mylib::)
#
#    install(FILES
#            "${CMAKE_CURRENT_BINARY_DIR}/mylib-config.cmake"
#            "${CMAKE_CURRENT_BINARY_DIR}/mylib-config-version.cmake"
#            COMPONENT mylib-dev
#            DESTINATION "${MYLIB_INSTALL_CMAKEDIR}")
#
#    if (MSVC)
#        set(pdb_file "")
#        set(pdb_file_destination "")
#
#        if (BUILD_SHARED_LIBS)
#            set(pdb_file "$<TARGET_PDB_FILE:mylib>")
#            set(pdb_file_destination "${CMAKE_INSTALL_BINDIR}")
#        else ()
#            # TARGET_PDB_FILE does not work for pdb file generated for static library build, determining it manually
#            set(pdb_file "$<TARGET_FILE_DIR:mylib>/$<TARGET_FILE_PREFIX:mylib>$<TARGET_FILE_BASE_NAME:mylib>.pdb")
#            set(pdb_file_destination "${CMAKE_INSTALL_LIBDIR}")
#        endif ()
#
#        install(FILES "${pdb_file}"
#                COMPONENT mylib-dev
#                CONFIGURATIONS Debug RelWithDebInfo
#                DESTINATION "${pdb_file_destination}"
#                OPTIONAL)
#    endif ()
#endif ()

#----------------------------------------------------------------------------------------------------------------------
# other targets
#----------------------------------------------------------------------------------------------------------------------

#if (MYLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
#endif ()

#if (MYLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
#endif ()

if (MYLIB_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs include)
endif ()
