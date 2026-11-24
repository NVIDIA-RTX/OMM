add_executable(omm-cpu-benchmark main.cpp)
target_link_libraries(omm-cpu-benchmark PRIVATE ${OMM_LIB_TARGET_NAME})
target_compile_definitions(omm-cpu-benchmark PRIVATE
    OMM_BENCHMARK_DEFAULT_INPUT="${PROJECT_SOURCE_DIR}/assets/omm_example_data/leaves.bin")
set_target_properties(omm-cpu-benchmark PROPERTIES FOLDER "Tools")
