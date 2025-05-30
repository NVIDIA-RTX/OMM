
if (WIN32)
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" ON)
 	set(EXTRA_LIBS nvrhi_d3d12 d3d12.lib dxgi.lib)
else()
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" OFF)
 	set(EXTRA_LIBS )
endif()

if (OMM_INTEGRATION_LAYER_NVRHI)
    set(OMM_LIB_NVRHI_HEADERS omm-gpu-nvrhi.h)
    set(OMM_LIB_NVRHI_SOURCE omm-gpu-nvrhi.cpp)
    add_library(omm-gpu-nvrhi STATIC ${OMM_LIB_NVRHI_HEADERS} ${OMM_LIB_NVRHI_SOURCE})
    target_include_directories(omm-gpu-nvrhi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(omm-gpu-nvrhi ${OMM_LIB_TARGET_NAME} nvrhi ${EXTRA_LIBS})
    set_target_properties(omm-gpu-nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}")
endif()

