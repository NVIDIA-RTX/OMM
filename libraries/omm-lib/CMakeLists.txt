# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
cmake_minimum_required(VERSION 3.12)

if (WIN32)
    set(DXIL_DEFAULT ON)
else()
    set(DXIL_DEFAULT OFF)
endif()

option(OMM_DISABLE_INTERPROCEDURAL_OPTIMIZATION "disable interprocedural optimization" OFF)
option(OMM_ENABLE_OPENMP "enable openmp" ON)
option(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL "Embedded precompiled DXIL shaders. Require path to dxc.exe (normally located in Window SDK)." ${DXIL_DEFAULT})
option(OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV "Embedded precompiled SPIRV shaders. Require path to Vulkan SDK." ON)
option(OMM_STATIC_LIBRARY "build static lib" OFF)
option(OMM_CROSSCOMPILE_AARCH64 "cross compilation for aarch64" OFF)
option(OMM_CROSSCOMPILE_X86_64 "cross compilation for x86_64" OFF)
option(OMM_SHADER_DEBUG_INFO "enable embedded shader debug info" OFF)
option(OMM_LIB_INSTALL "Generate install rules for OMM" ON)
option(OMM_ENABLE_FAST_MATH "Enable fast math optimizations()" ON)

if (OMM_ENABLE_OPENMP)
find_package(OpenMP)
endif()

if (OMM_CROSSCOMPILE_AARCH64)
    set(CMAKE_SYSTEM_PROCESSOR "aarch64")
    message(STATUS "CROSSCOMPILE_AARCH64 enabled.")
endif()
if (OMM_CROSSCOMPILE_X86_64)
    set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    message(STATUS "CROSSCOMPILE_X86_64 enabled.")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POSITION_INDEPENDENT_BINARIES ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if (OMM_DISABLE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if ("${OMM_SHADER_OUTPUT_PATH}" STREQUAL "")
    set(OMM_SHADER_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/shaders")
endif()

# Vulkan resource offsets
set(OMM_VK_S_SHIFT 100 CACHE STRING "OMM_VK_S_SHIFT")
set(OMM_VK_T_SHIFT 200 CACHE STRING "OMM_VK_T_SHIFT")
set(OMM_VK_B_SHIFT 300 CACHE STRING "OMM_VK_B_SHIFT")
set(OMM_VK_U_SHIFT 400 CACHE STRING "OMM_VK_U_SHIFT")
add_definitions(-DOMM_VK_S_SHIFT=${OMM_VK_S_SHIFT} -DOMM_VK_T_SHIFT=${OMM_VK_T_SHIFT} -DOMM_VK_B_SHIFT=${OMM_VK_B_SHIFT} -DOMM_VK_U_SHIFT=${OMM_VK_U_SHIFT})

if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL OR OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    include(ShaderCompilation.cmake)
endif()

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(OMM_IS_SUBMODULE OFF)
else()
    set(OMM_IS_SUBMODULE ON)
endif()

# list C/C++ files and add them to OMMBake project
file(GLOB OMM_SOURCE "src/*.cpp" "src/*.h" "src/*.hpp")
source_group("Sources" FILES ${OMM_SOURCE})
file(GLOB OMM_RESOURCE "src/*.rc")
source_group("Sources" FILES ${OMM_RESOURCE})
file(GLOB OMM_HEADERS "include/*")
source_group("Include" FILES ${OMM_HEADERS})

if (OMM_STATIC_LIBRARY)
    add_library(${OMM_LIB_TARGET_NAME} STATIC ${OMM_SOURCE} ${OMM_RESOURCE} ${OMM_HEADERS})
else()
    add_library(${OMM_LIB_TARGET_NAME} SHARED ${OMM_SOURCE} ${OMM_RESOURCE} ${OMM_HEADERS})
endif()

if(OMM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${OMM_LIB_TARGET_NAME} OpenMP::OpenMP_CXX)
    else()
        message(STATUS "OpenMP not found. OpenMP will be disabled")
    endif()
endif()

target_link_libraries(${OMM_LIB_TARGET_NAME} glm stb_lib xxHash::xxhash lz4) 

set_target_properties(${OMM_LIB_TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
target_include_directories(${OMM_LIB_TARGET_NAME} PUBLIC "include")
target_include_directories(${OMM_LIB_TARGET_NAME} PRIVATE "shaders" "${CMAKE_CURRENT_SOURCE_DIR}/src")
set_property(TARGET ${OMM_LIB_TARGET_NAME} PROPERTY FOLDER "${OMM_PROJECT_FOLDER}") 

if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
    add_dependencies(${OMM_LIB_TARGET_NAME} ${OMM_LIB_TARGET_NAME}-dxil)
endif()

if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    add_dependencies(${OMM_LIB_TARGET_NAME} ${OMM_LIB_TARGET_NAME}-spirv)
endif()

if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL OR OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    target_include_directories(${OMM_LIB_TARGET_NAME} PRIVATE "${OMM_SHADER_OUTPUT_PATH}")
endif()
if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
    target_compile_definitions(${OMM_LIB_TARGET_NAME} PRIVATE OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
endif()
if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    target_compile_definitions(${OMM_LIB_TARGET_NAME} PRIVATE OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
endif()

if(WIN32)
    target_compile_definitions(${OMM_LIB_TARGET_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE)
else()
    # on non-Windows, FXC is not available and VulkanSDK DXC does not produce signed DXIL
    target_compile_definitions(${OMM_LIB_TARGET_NAME} PRIVATE OMM_ONLY_SPIRV_SHADERS_AVAILABLE=1)
endif()

if (OMM_STATIC_LIBRARY)
    target_compile_definitions(${OMM_LIB_TARGET_NAME} PUBLIC "OMM_API=extern \"C\"")
else()
    if(WIN32)
        target_compile_definitions(${OMM_LIB_TARGET_NAME} PRIVATE "OMM_API=extern \"C\" __declspec(dllexport)")
    else()
        target_compile_definitions(${OMM_LIB_TARGET_NAME} PRIVATE "OMM_API=extern \"C\" __attribute__((visibility(\"default\")))")
    endif()
endif()

if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(OMM_ARCHITECTURE_COMPILE_OPTIONS -msse4.1)
endif()

if (OMM_ENABLE_FAST_MATH)
    # Only enable fast math on msvc right now. It works on other compilers but requires a thurogh analysis of the consequecnes 
    #set(OMM_COMPILER_COMPILE_OPTIONS_CLANG ${OMM_COMPILER_COMPILE_OPTIONS_CLANG} -ffast-math)
    #set(OMM_COMPILER_COMPILE_OPTIONS_GCC ${OMM_COMPILER_COMPILE_OPTIONS_GCC} -ffast-math)
    set(OMM_COMPILER_COMPILE_OPTIONS_MSVC ${OMM_COMPILER_COMPILE_OPTIONS_MSVC} /fp:fast)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${OMM_LIB_TARGET_NAME} PRIVATE ${OMM_ARCHITECTURE_COMPILE_OPTIONS} ${OMM_COMPILER_COMPILE_OPTIONS_CLANG} -Wno-missing-braces -Wno-return-type-c-linkage)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${OMM_LIB_TARGET_NAME} PRIVATE ${OMM_ARCHITECTURE_COMPILE_OPTIONS} ${OMM_COMPILER_COMPILE_OPTIONS_GCC})
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${OMM_LIB_TARGET_NAME} PRIVATE ${OMM_COMPILER_COMPILE_OPTIONS_MSVC})
endif()

if (NOT OMM_IS_SUBMODULE)
    set(OMM_OUTPUT_BIN_PATH "${CMAKE_SOURCE_DIR}/Bin")
else()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${OMM_LIB_TARGET_NAME})
endif()

set_target_properties(${OMM_LIB_TARGET_NAME} PROPERTIES FOLDER ${OMM_PROJECT_FOLDER})

if (NOT ${OMM_OUTPUT_BIN_PATH} STREQUAL "")
    message(STATUS "OMM-SDK output path: '${OMM_OUTPUT_BIN_PATH}'")
    set_target_properties(${OMM_LIB_TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OMM_OUTPUT_BIN_PATH}/$<CONFIG>")
    set_target_properties(${OMM_LIB_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OMM_OUTPUT_BIN_PATH}/$<CONFIG>")
    set_target_properties(${OMM_LIB_TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OMM_OUTPUT_BIN_PATH}/$<CONFIG>")
endif()

if (OMM_LIB_INSTALL)
    install (TARGETS ${OMM_LIB_TARGET_NAME}
             ARCHIVE DESTINATION lib
             LIBRARY DESTINATION lib
             RUNTIME DESTINATION bin)

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/omm.h" DESTINATION include)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/omm.hpp" DESTINATION include)
endif()
