cmake_minimum_required(VERSION 3.0)
project(models_ops)
execute_process (
    COMMAND bash -c "python3 -c 'from oneflow.framework.sysconfig import get_include; print(get_include(), end=\"\")'"
    OUTPUT_VARIABLE ONEFLOW_INCLUDE_PATH
)

execute_process (
    COMMAND bash -c "python3 -c 'from oneflow.framework.sysconfig import get_compile_flags; print(\" \".join(get_compile_flags()), end=\"\")'"
    OUTPUT_VARIABLE ONEFLOW_COMPILE_FLAGS
)

execute_process (
    COMMAND bash -c "python3 -c 'from oneflow.framework.sysconfig import get_lib; print(get_lib(), end=\"\")'"
    OUTPUT_VARIABLE ONEFLOW_LIB_PATH
)

find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CUDA_HOST_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xcompiler -Wextra --disable-warnings -DWITH_CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ONEFLOW_COMPILE_FLAGS} -O3 -g -std=c++11 -Wall -Wno-sign-compare -Wno-unused-function -fPIC")
set(CUDA_VERBOSE_BUILD OFF)

file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
file(GLOB_RECURSE CUDASRC "*.cu")
file(GLOB_RECURSE CUDAHDR "*.cuh")
file(GLOB_RECURSE SO ${ONEFLOW_LIB_PATH}/_oneflow_internal.*.so)
list(APPEND SRC ${HDR})
list(APPEND SRC ${CUDASRC})
list(APPEND SRC ${CUDAHDR})

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../../)
target_include_directories(${PROJECT_NAME} PUBLIC ${ONEFLOW_INCLUDE_PATH})
add_library(oneflow_so SHARED IMPORTED)
set_property(TARGET oneflow_so PROPERTY IMPORTED_LOCATION ${SO})
target_link_libraries(${PROJECT_NAME} oneflow_so)

add_custom_target(code_format)
foreach(source_file ${SRC})
    add_custom_command(TARGET code_format PRE_BUILD
    COMMAND clang-format -i -style=file ${source_file})
endforeach()
add_dependencies(${PROJECT_NAME} code_format)
