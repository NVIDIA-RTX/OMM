
option(NVRHI_INSTALL OFF)
set (SHADERMAKE_BIN_OUTPUT_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" CACHE STRING "")
option(BUILD_SHARED_LIBS "" OFF)
option(LZ4_BUILD_CLI "" OFF)

# donut & nvrhi
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/donut AND OMM_BUILD_VIEWER)
	option(DONUT_WITH_AUDIO "" OFF)
	option(DONUT_WITH_ASSIMP "" OFF)
	option(DONUT_WITH_DX11 "" OFF)
	option(DONUT_WITH_LZ4 "" OFF)
	option(DONUT_WITH_MINIZ "" OFF)
	option(DONUT_WITH_STATIC_SHADERS "" ON)
	option(DONUT_WITH_TASKFLOW "" OFF)
	option(DONUT_WITH_TINYEXR "" OFF)
	option(DONUT_WITH_STREAMLINE "" OFF)
	
	set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework")
	add_subdirectory(donut)

	set_property(TARGET donut_app PROPERTY FOLDER "External/Donut") 
	set_property(TARGET donut_core PROPERTY FOLDER "External/Donut") 
	set_property(TARGET donut_engine PROPERTY FOLDER "External/Donut") 
	set_property(TARGET donut_render PROPERTY FOLDER "External/Donut") 
	set_property(TARGET donut_shaders PROPERTY FOLDER "External/Donut") 
	set_property(TARGET glfw PROPERTY FOLDER "External/GLFW3")
	set_property(TARGET update_mappings PROPERTY FOLDER "External/GLFW3") 

	set_property(TARGET imgui PROPERTY FOLDER "External")
	set_property(TARGET jsoncpp_static PROPERTY FOLDER "External")
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/donut/nvrhi AND OMM_INTEGRATION_LAYER_NVRHI)
	if(NOT TARGET nvrhi)
		option(NVRHI_INSTALL OFF)
		set(NVRHI_WITH_SHADER_COMPILER OFF CACHE BOOL "" FORCE)
		set(NVRHI_WITH_DX11 OFF CACHE BOOL "" FORCE)
		set(NVRHI_WITH_VULKAN OFF CACHE BOOL "" FORCE)
		add_subdirectory(donut/nvrhi)
		target_include_directories(nvrhi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/donut)
	endif()
endif()

if(TARGET DirectX-Guids)
	set_property(TARGET DirectX-Guids PROPERTY FOLDER "External/DirectX-Guids")
endif()
if(TARGET DirectX-Headers)
	set_property(TARGET DirectX-Headers PROPERTY FOLDER "External/DirectX-Headers") 
endif()

# GLM
add_subdirectory("glm")
set_property(TARGET glm PROPERTY FOLDER "External") 

if(BUILD_STATIC_LIBS)
	set_property(TARGET glm_static PROPERTY FOLDER "External") 
endif()

if(BUILD_SHARED_LIBS)
	set_property(TARGET glm_shared PROPERTY FOLDER "External") 
endif()

# lz4
set(LZ4_BUILD_CLI OFF)
set(LZ4_BUILD_LEGACY_LZ4C OFF)
add_subdirectory("lz4/build/cmake")
set_property(TARGET lz4_static PROPERTY FOLDER "External") 

# ShaderMake
if (NOT TARGET ShaderMake)
    add_subdirectory("ShaderMake")
    set_property(TARGET ShaderMake PROPERTY FOLDER "External") 
    set_property(TARGET ShaderMakeBlob PROPERTY FOLDER "External")
endif()

# xxHash
set(XXHASH_BUILD_XXHSUM OFF)
add_subdirectory("xxHash/build/cmake")
set_property(TARGET xxhash PROPERTY FOLDER "External") 

# stb
add_library(stb_lib INTERFACE)
target_include_directories(stb_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stb>)

# lz4
if (TARGET lz4)
	set_property(TARGET lz4 PROPERTY FOLDER "External") 
endif()
if (TARGET lz4_static)
	set_property(TARGET lz4_static PROPERTY FOLDER "External") 
endif()

if (TARGET ShaderMake)
    set_property(TARGET ShaderMake PROPERTY FOLDER "External") 
    set_property(TARGET ShaderMakeBlob PROPERTY FOLDER "External")
endif()

# nvrhi
if (TARGET nvrhi)
	set_property(TARGET nvrhi PROPERTY FOLDER "External")
endif()

if (TARGET nvrhi_d3d12)
	set_property(TARGET nvrhi_d3d12 PROPERTY FOLDER "External")
endif()

if (TARGET nvrhi_vk)
	set_property(TARGET nvrhi_vk PROPERTY FOLDER "External") 
endif()

# gtest
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gtest AND OMM_ENABLE_TESTS)
	set(BUILD_GMOCK OFF)
	set(INSTALL_GTEST OFF)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory("gtest")
	set_target_properties("gtest_main" "gtest" PROPERTIES FOLDER "External")
endif()