# DXC on Windows does not like forward slashes
if (WIN32)
    string(REPLACE "/" "\\" OMM_SHADER_INCLUDE_PATH "${OMM_SHADER_INCLUDE_PATH}")
    string(REPLACE "/" "\\" OMM_HEADER_INCLUDE_PATH "${OMM_HEADER_INCLUDE_PATH}")
endif()

# Find DXC
if (WIN32)

    # DXIL
    if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
        if (DEFINED CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
            set (OMM_WINDOWS_SDK_VERSION ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION})
        elseif (DEFINED ENV{WindowsSDKLibVersion})
            string (REGEX REPLACE "\\\\$" "" OMM_WINDOWS_SDK_VERSION "$ENV{WindowsSDKLibVersion}")
        else()
            message(FATAL_ERROR "WindowsSDK is not installed. (CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION is not defined; WindowsSDKLibVersion is '$ENV{WindowsSDKLibVersion}')")
        endif()

        get_filename_component(OMM_WINDOWS_SDK_ROOT
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE)

        set(OMM_WINDOWS_SDK_BIN "${OMM_WINDOWS_SDK_ROOT}/bin/${OMM_WINDOWS_SDK_VERSION}/x64")

        find_program(OMM_DXC_PATH "${OMM_WINDOWS_SDK_BIN}/dxc")

        if (OMM_DXC_PATH)
            message(STATUS "DXIL dxc.exe was found in ${OMM_DXC_PATH}" )
        else()
            message(FATAL_ERROR "DXIL dxc.exe was not found on the system. Is the Windows SDK installed? Resolve this by either:
            1. Disable embedded DXIL shaders OMM_ENABLE_PRECOMPILED_SHADERS_DXIL OFF
            2. Provide a custom path to DXIL dxc.exe (OMM_DXC_PATH)
            3. Install Windows SDK on the system\n")
        endif()

        message(STATUS "DXIL will be generated by: ${OMM_DXC_PATH}")

    endif()

    # SPIRV
    if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
        
        find_program(OMM_VULKAN_DXC_SPIRV_PATH "$ENV{VULKAN_SDK}/Bin/dxc")

        if (OMM_VULKAN_DXC_SPIRV_PATH)
            message(STATUS "SPIRV dxc.exe was found in ${OMM_VULKAN_DXC_SPIRV_PATH}" )
        else()
            message(FATAL_ERROR "SPIRV dxc.exe was not found on the system. Is the Vulkan SDK installed? Resolve this by either:
            1. Disable embedded SPIRV shaders OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV OFF
            2. Provide a custom path to Vulkan dxc.exe (OMM_VULKAN_DXC_SPIRV_PATH)
            3. Install Vulkan SDK on the system and make sure environment variable VULKAN_SDK is set.\n")
        endif()

        message(STATUS "SPIRV will be generated by: ${OMM_VULKAN_DXC_SPIRV_PATH}")
    endif()
else()
    message(WARNING "Shader compilation not supported on the current platform.")
    set(OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV OFF CACHE BOOL "" FORCE)
    set(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL OFF CACHE BOOL "" FORCE)
endif()

function(get_shader_profile_from_name FILE_NAME DXC_PROFILE)
    get_filename_component(EXTENSION ${FILE_NAME} EXT)
    if ("${EXTENSION}" STREQUAL ".cs.hlsl")
        set(DXC_PROFILE "cs_6_0" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".vs.hlsl")
        set(DXC_PROFILE "vs_6_0" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".gs.hlsl")
        set(DXC_PROFILE "gs_6_0" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".ps.hlsl")
        set(DXC_PROFILE "ps_6_0" PARENT_SCOPE)
    endif()
endfunction()

macro(list_hlsl_headers OMM_HLSL_FILES OMM_HEADER_FILES)
    foreach(FILE_NAME ${OMM_HLSL_FILES})
        set(DXC_PROFILE "")
        get_shader_profile_from_name(${FILE_NAME} DXC_PROFILE)
        if ("${DXC_PROFILE}" STREQUAL "")
            list(APPEND OMM_HEADER_FILES ${FILE_NAME})
            set_source_files_properties(${FILE_NAME} PROPERTIES VS_TOOL_OVERRIDE "None")
        endif()
    endforeach() 
endmacro()

set (OMM_DXC_VK_SHIFTS 
    -fvk-s-shift ${OMM_VK_S_SHIFT} 0 -fvk-s-shift ${OMM_VK_S_SHIFT} 1 -fvk-s-shift ${OMM_VK_S_SHIFT} 2
    -fvk-t-shift ${OMM_VK_T_SHIFT} 0 -fvk-t-shift ${OMM_VK_T_SHIFT} 1 -fvk-t-shift ${OMM_VK_T_SHIFT} 2
    -fvk-b-shift ${OMM_VK_B_SHIFT} 0 -fvk-b-shift ${OMM_VK_B_SHIFT} 1 -fvk-b-shift ${OMM_VK_B_SHIFT} 2
    -fvk-u-shift ${OMM_VK_U_SHIFT} 0 -fvk-u-shift ${OMM_VK_U_SHIFT} 1 -fvk-u-shift ${OMM_VK_U_SHIFT} 2)

if (OMM_SHADER_DEBUG_INFO)
set (DXC_ADDITIONAL_OPTIONS -Qembed_debug -Zi)
endif()

macro(list_hlsl_shaders OMM_HLSL_FILES OMM_HEADER_FILES OMM_SHADER_FILES)
    foreach(FILE_NAME ${OMM_HLSL_FILES})
        get_filename_component(NAME_ONLY ${FILE_NAME} NAME)
        string(REGEX REPLACE "\\.[^.]*$" "" NAME_ONLY ${NAME_ONLY})
        string(REPLACE "." "_" BYTECODE_ARRAY_NAME "${NAME_ONLY}")
        set(DXC_PROFILE "")
        set(OUTPUT_PATH_DXIL "${OMM_SHADER_OUTPUT_PATH}/${NAME_ONLY}.dxil")
        set(OUTPUT_PATH_SPIRV "${OMM_SHADER_OUTPUT_PATH}/${NAME_ONLY}.spirv")
        get_shader_profile_from_name(${FILE_NAME} DXC_PROFILE)

        # add DXC compilation step (DXIL)
        if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
            if (NOT "${DXC_PROFILE}" STREQUAL "" AND NOT "${OMM_DXC_PATH}" STREQUAL "")
                add_custom_command(
                        OUTPUT ${OUTPUT_PATH_DXIL} ${OUTPUT_PATH_DXIL}.h
                        COMMAND ${OMM_DXC_PATH} -E main -DCOMPILER_DXC=1 -T ${DXC_PROFILE} -I "${OMM_HEADER_INCLUDE_PATH}" ${DXC_ADDITIONAL_OPTIONS}
                            -I "${OMM_SHADER_INCLUDE_PATH}" ${FILE_NAME}
                            -Vn g_${BYTECODE_ARRAY_NAME}_dxil -Fh ${OUTPUT_PATH_DXIL}.h -Fo ${OUTPUT_PATH_DXIL}
                        MAIN_DEPENDENCY ${FILE_NAME}
                        DEPENDS ${OMM_HEADER_FILES}
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                        VERBATIM
                )
                list(APPEND OMM_SHADER_FILES ${OUTPUT_PATH_DXIL} ${OUTPUT_PATH_DXIL}.h)
            endif()
        endif()
        # add one more DXC compilation step (SPIR-V)
        if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
            if (NOT "${DXC_PROFILE}" STREQUAL "")
                add_custom_command(
                        OUTPUT ${OUTPUT_PATH_SPIRV} ${OUTPUT_PATH_SPIRV}.h
                        COMMAND ${OMM_VULKAN_DXC_SPIRV_PATH} -E main -DCOMPILER_DXC=1 -DVULKAN=1 -T ${DXC_PROFILE}
                            -I "${OMM_HEADER_INCLUDE_PATH}" ${DXC_ADDITIONAL_OPTIONS} -I "${OMM_SHADER_INCLUDE_PATH}"
                            ${FILE_NAME} -spirv -Vn g_${BYTECODE_ARRAY_NAME}_spirv -Fh ${OUTPUT_PATH_SPIRV}.h
                            -Fo ${OUTPUT_PATH_SPIRV} ${OMM_DXC_VK_SHIFTS} -fspv-target-env=vulkan1.1
                        MAIN_DEPENDENCY ${FILE_NAME}
                        DEPENDS ${OMM_HEADER_FILES}
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                        VERBATIM
                )
                list(APPEND OMM_SHADER_FILES ${OUTPUT_PATH_SPIRV} ${OUTPUT_PATH_SPIRV}.h)
            endif()
        endif()
    endforeach()
endmacro()