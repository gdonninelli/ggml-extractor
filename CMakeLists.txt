cmake_minimum_required(VERSION 3.20)
project(ggml-extractor VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLAMA_CPP_DIR "" CACHE PATH "Path to a llama.cpp source checkout")
if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to a llama.cpp source checkout "
                        "(-DLLAMA_CPP_DIR=/path/to/llama.cpp)")
endif()

# Keep the dependency surface minimal: no server, tools, or tests from llama.cpp.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

find_package(Threads REQUIRED)

add_library(ggml_extractor STATIC
    src/extractor.cpp
    src/npy_writer.cpp
    src/llama_session.cpp
)
target_include_directories(ggml_extractor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(ggml_extractor PUBLIC cxx_std_17)
target_link_libraries(ggml_extractor PUBLIC llama Threads::Threads)

if(MSVC)
    target_compile_options(ggml_extractor PRIVATE /W4)
else()
    target_compile_options(ggml_extractor PRIVATE -Wall -Wextra -Wpedantic)
endif()

foreach(example prefill batch generate raw_ggml)
    if(example STREQUAL "prefill")
        set(src examples/01_prefill.cpp)
        set(tgt prefill_example)
    elseif(example STREQUAL "batch")
        set(src examples/02_batch.cpp)
        set(tgt batch_example)
    elseif(example STREQUAL "generate")
        set(src examples/03_generate.cpp)
        set(tgt generate_example)
    else()
        set(src examples/04_raw_ggml.cpp)
        set(tgt raw_ggml_example)
    endif()
    add_executable(${tgt} ${src})
    target_link_libraries(${tgt} PRIVATE ggml_extractor)
    set_target_properties(${tgt} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ggml_extractor ARCHIVE DESTINATION lib)
