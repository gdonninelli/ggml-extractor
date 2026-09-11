cmake_minimum_required(VERSION 3.20)
project(ggml-extractor VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLAMA_CPP_DIR "" CACHE PATH "Path to a llama.cpp source checkout")
if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to a llama.cpp source checkout "
                        "(-DLLAMA_CPP_DIR=/path/to/llama.cpp)")
endif()

option(GGML_EXTRACTOR_MULTIMODAL "Build vision/audio support (llama.cpp libmtmd)" ON)

# Keep the dependency surface minimal: no server, tools, or tests from llama.cpp.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)

# libmtmd lives under llama.cpp's tools/ tree but ships a standalone build hook
# that skips common/ and the CLI binaries.
if(GGML_EXTRACTOR_MULTIMODAL)
    set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

find_package(Threads REQUIRED)

set(GGML_EXTRACTOR_SOURCES
    src/backend.cpp
    src/capture.cpp
    src/model.cpp
    src/npy_writer.cpp
    src/session.cpp
)
if(GGML_EXTRACTOR_MULTIMODAL)
    list(APPEND GGML_EXTRACTOR_SOURCES src/multimodal.cpp)
endif()

add_library(ggml_extractor STATIC ${GGML_EXTRACTOR_SOURCES})
target_include_directories(ggml_extractor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(ggml_extractor PUBLIC cxx_std_17)
target_link_libraries(ggml_extractor PUBLIC llama Threads::Threads)
if(GGML_EXTRACTOR_MULTIMODAL)
    target_link_libraries(ggml_extractor PUBLIC mtmd)
    target_compile_definitions(ggml_extractor PUBLIC GGML_EXTRACTOR_MULTIMODAL)
endif()

if(MSVC)
    target_compile_options(ggml_extractor PRIVATE /W4)
else()
    target_compile_options(ggml_extractor PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 05_service.cpp compiles either way: its image stages are guarded by the
# GGML_EXTRACTOR_MULTIMODAL define the library exports.
set(GGML_EXTRACTOR_EXAMPLES
    prefill_example  examples/01_prefill.cpp
    batch_example    examples/02_batch.cpp
    generate_example examples/03_generate.cpp
    raw_ggml_example examples/04_raw_ggml.cpp
    service_example  examples/05_service.cpp
)

list(LENGTH GGML_EXTRACTOR_EXAMPLES _example_entries)
math(EXPR _example_last "${_example_entries} / 2 - 1")
foreach(index RANGE ${_example_last})
    math(EXPR _target_index "${index} * 2")
    math(EXPR _source_index "${index} * 2 + 1")
    list(GET GGML_EXTRACTOR_EXAMPLES ${_target_index} _target)
    list(GET GGML_EXTRACTOR_EXAMPLES ${_source_index} _source)
    add_executable(${_target} ${_source})
    target_link_libraries(${_target} PRIVATE ggml_extractor)
    set_target_properties(${_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ggml_extractor ARCHIVE DESTINATION lib)
