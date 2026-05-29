cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME "hello_cuda")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CUDA_TEST_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_executable(${PROJECT_NAME} main.cu)

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /utf-8>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/W4 -Xcompiler=/utf-8>
)

if(CUDA_TEST_WARNINGS_AS_ERRORS)
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/WX>
  )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_ARCHITECTURES native
)
