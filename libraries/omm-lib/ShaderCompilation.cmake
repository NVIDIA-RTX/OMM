# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

##########################################################################
###################### Shader Compilation Util ###########################
##########################################################################

function(compile_shaders)

    if (OMM_SHADER_DEBUG_INFO)
    set (DXC_ADDITIONAL_OPTIONS -Qembed_debug -Zi)
    endif()

    set(options "")
    set(oneValueArgs TARGET CONFIG FOLDER DXIL SPIRV DXBC SPIRV_DXC CFLAGS INCLUDE)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(params "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT params_TARGET)
        message(FATAL_ERROR "compile_shaders: TARGET argument missing")
    endif()
    if (NOT params_CONFIG)
        message(FATAL_ERROR "compile_shaders: CONFIG argument missing")
    endif()

    # just add the source files to the project as documents, they are built by the script
    set_source_files_properties(${params_SOURCES} PROPERTIES VS_TOOL_OVERRIDE "None") 

    add_custom_target(${params_TARGET}
        DEPENDS ShaderMake
        SOURCES ${params_SOURCES})

    if (NOT params_INCLUDE)
        set(INCLUDE_PATH "")
    else()
        set(INCLUDE_PATH -I ${params_INCLUDE})
    endif()

    if (params_DXIL)
        if (NOT SHADERMAKE_DXC_PATH)
            message(FATAL_ERROR "compile_shaders: DXC not found --- please set SHADERMAKE_DXC_PATH to the full path to the DXC binary")
        endif()

        if (NOT params_CFLAGS)
            set(CFLAGS "-Zi -Qembed_debug -O3 -WX")
        else()
            set(CFLAGS ${params_CFLAGS})
        endif()

        add_custom_command(TARGET ${params_TARGET} PRE_BUILD
                          COMMAND ShaderMake
                                   --header
                                   --config=${params_CONFIG}
                                   --out ${params_DXIL}
                                   --compiler ${SHADERMAKE_DXC_PATH}
                                   --platform=DXIL
                                   ${INCLUDE_PATH}
                                   )
    endif()

    if (params_SPIRV)
        if (NOT SHADERMAKE_DXC_VK_PATH)
            message(FATAL_ERROR "compile_shaders: DXC for SPIR-V not found --- please set SHADERMAKE_DXC_VK_PATH to the full path to the DXC binary")
        endif()

        if (NOT params_CFLAGS)
            set(CFLAGS "$<IF:$<CONFIG:Debug>,-Zi,> -fspv-target-env=vulkan1.2 -O3 -WX")
        else()
            set(CFLAGS ${params_CFLAGS})
        endif()

        add_custom_command(TARGET ${params_TARGET} PRE_BUILD
                            COMMAND ShaderMake
                                    --header
                                    --vulkanVersion=1.2
                                    --config=${params_CONFIG}
                                    --out ${params_SPIRV}
                                    --compiler ${SHADERMAKE_DXC_VK_PATH}
                                    --platform=SPIRV
                                    --sRegShift=${OMM_VK_S_SHIFT}
                                    --tRegShift=${OMM_VK_T_SHIFT}
                                    --bRegShift=${OMM_VK_B_SHIFT}
                                    --uRegShift=${OMM_VK_U_SHIFT}
                                    ${INCLUDE_PATH}
                                    )
    endif()

    if(params_FOLDER)
        set_target_properties(${params_TARGET} PROPERTIES FOLDER ${params_FOLDER})
    endif()
endfunction()

function(util_get_shader_profile_from_name FILE_NAME DXC_PROFILE)
    get_filename_component(EXTENSION ${FILE_NAME} EXT)
    if ("${EXTENSION}" STREQUAL ".cs.hlsl")
        set(DXC_PROFILE "cs" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".vs.hlsl")
        set(DXC_PROFILE "vs" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".gs.hlsl")
        set(DXC_PROFILE "gs" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".ps.hlsl")
        set(DXC_PROFILE "ps" PARENT_SCOPE)
    endif()
endfunction()

function(util_generate_shader_config_file OUT_FILE_NAME DIR DEFINES)
    file(GLOB_RECURSE HLSL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/*.hlsl")

    set(out_content "")
    foreach(FILE_NAME ${HLSL_FILES})
        get_filename_component(NAME_ONLY ${FILE_NAME} NAME)
        util_get_shader_profile_from_name(${FILE_NAME} DXC_PROFILE)
        set(out_content "${out_content}${DIR}/${NAME_ONLY} -T ${DXC_PROFILE} -E main ${DEFINES}\n")
    endforeach()

    file(WRITE ${OUT_FILE_NAME} ${out_content})
endfunction()

# Let CMake generate the shaders.cfg file 
util_generate_shader_config_file(
    "shaders.cfg" 
    "shaders"
    ""
)

if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
    compile_shaders(
        TARGET ${OMM_LIB_TARGET_NAME}-dxil
        CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        FOLDER ${OMM_PROJECT_FOLDER}
        DXIL ${PROJECT_BINARY_DIR}/bin
    )
    set_target_properties(${OMM_LIB_TARGET_NAME}-dxil PROPERTIES FOLDER ${OMM_PROJECT_FOLDER}/Shaders)
endif()

if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    compile_shaders(
        TARGET ${OMM_LIB_TARGET_NAME}-spirv
        CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        FOLDER ${OMM_PROJECT_FOLDER}
        SPIRV ${PROJECT_BINARY_DIR}/bin
    )
    set_target_properties(${OMM_LIB_TARGET_NAME}-spirv PROPERTIES FOLDER ${OMM_PROJECT_FOLDER}/Shaders)
endif()
