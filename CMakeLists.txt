cmake_minimum_required(VERSION 4.1)

option(OMM_BUILD_BENCHMARK "Build the CPU bake benchmark" ON)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/gtest)
	option(OMM_ENABLE_TESTS "Enable unit test" OFF)
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/donut)
	option(OMM_BUILD_VIEWER "Build omm viewer tool" ON)
else()
	option(OMM_BUILD_VIEWER "Build omm viewer tool" OFF)
endif()

if (NOT OMM_LIB_TARGET_NAME)
    option(OMM_USE_LEGACY_OMM_LIB_NAME "Use the legacy target name of omm-lib: \"omm-sdk\"" OFF)

    if (OMM_USE_LEGACY_OMM_LIB_NAME)
	    set(OMM_LIB_TARGET_NAME omm-sdk)
    else()
        set(OMM_LIB_TARGET_NAME omm-lib)
    endif()
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/libraries/omm-lib/src/version.h" ver_h)
string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver_h})
set(ver_major ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver_h})
set(ver_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_BUILD ([0-9]*)" _ ${ver_h})
set(ver_patch ${CMAKE_MATCH_1})

project("Opacity Micro-Map SDK" VERSION ${ver_major}.${ver_minor}.${ver_patch} LANGUAGES CXX)
message(STATUS "${PROJECT_NAME} version: ${ver_major}.${ver_minor}.${ver_patch}")

if (WIN32)
    option(OMM_ENABLE_GPU_TESTS "Build GPU tests" ${OMM_ENABLE_TESTS})
    if (OMM_BUILD_VIEWER OR OMM_ENABLE_GPU_TESTS)
        set(OMM_NVRHI_INTEGRATION_DEFAULT ON)
    else()
        set(OMM_NVRHI_INTEGRATION_DEFAULT OFF)
    endif()
    option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" ${OMM_NVRHI_INTEGRATION_DEFAULT})
else()
    option(OMM_ENABLE_GPU_TESTS "Build GPU tests" OFF)
    option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" OFF)
endif()

option(OMM_GPU_REFERENCE_MODE "Use precise, unoptimized software GPU math and sampling" OFF)
set(OMM_ENABLE_PRECISE_GPU_MATH ${OMM_GPU_REFERENCE_MODE})
set(OMM_DISABLE_GPU_SHADER_OPTIMIZATIONS ${OMM_GPU_REFERENCE_MODE})

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
message(STATUS "${CMAKE_BINARY_DIR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Windows: if install locations not set by user, set install prefix to "<build_dir>\install".
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
   
if (NOT DEFINED OMM_PROJECT_FOLDER)
    set(OMM_PROJECT_FOLDER "Libraries")
endif()

add_subdirectory(external)

if (TARGET nvrhi) # Must be after including External folder (but definition can also come from parent project.)
	option(OMM_BUILD_OMM_GPU_NVRHI "Build omm gpu nvrhi layer" ON)
else()
	option(OMM_BUILD_OMM_GPU_NVRHI "Build omm gpu nvrhi layer" OFF)
endif()

if (OMM_GPU_REFERENCE_MODE)
    message(STATUS "GPU reference mode uses precise, unoptimized software sqrt, division, and texture sampling")
endif()
if (OMM_ENABLE_TESTS AND OMM_ENABLE_GPU_TESTS)
    if (NOT WIN32 OR NOT OMM_BUILD_OMM_GPU_NVRHI OR NOT OMM_INTEGRATION_LAYER_NVRHI OR NOT TARGET nvrhi_d3d12)
        message(WARNING "GPU tests require Windows and the NVRHI D3D12 integration. Disabling OMM_ENABLE_GPU_TESTS.")
        set(OMM_ENABLE_GPU_TESTS OFF)
    else()
        if (NOT OMM_GPU_REFERENCE_MODE)
            message(WARNING
                "GPU tests are enabled without OMM_GPU_REFERENCE_MODE. "
                "Enable OMM_GPU_REFERENCE_MODE for accurate comparison with the fixed compute reference values. "
                "The GpuTest.ReferenceMode check will fail; remaining tests will run with the selected shader settings.")
        endif()
        set(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL ON)
    endif()
endif()

add_subdirectory(libraries/omm-lib)

if (OMM_BUILD_OMM_GPU_NVRHI)
    add_subdirectory(libraries/omm-gpu-nvrhi)
endif()

if (OMM_BUILD_VIEWER)
    add_subdirectory(tools/viewer)
endif()

if (OMM_BUILD_BENCHMARK)
    add_subdirectory(tools/benchmark)
endif()

if (OMM_ENABLE_TESTS)
    add_subdirectory(support/tests)
endif()

add_subdirectory(support/scripts)

if (OMM_BUILD_VIEWER)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT viewer)
elseif(OMM_ENABLE_TESTS)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT tests)
endif()
