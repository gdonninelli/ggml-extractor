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

# Prompt rendering uses llama.cpp's own Jinja engine, compiled straight from
# the checkout (no common/ dependency). Fail early if a checkout predating
# the engine's move to common/jinja is used.
set(GGML_EXTRACTOR_JINJA_SOURCES
    "${LLAMA_CPP_DIR}/common/jinja/lexer.cpp"
    "${LLAMA_CPP_DIR}/common/jinja/parser.cpp"
    "${LLAMA_CPP_DIR}/common/jinja/runtime.cpp"
    "${LLAMA_CPP_DIR}/common/jinja/string.cpp"
    "${LLAMA_CPP_DIR}/common/jinja/value.cpp"
    "${LLAMA_CPP_DIR}/common/unicode.cpp"
)
foreach(jinja_source IN LISTS GGML_EXTRACTOR_JINJA_SOURCES)
    if(NOT EXISTS "${jinja_source}")
        message(FATAL_ERROR "Missing ${jinja_source}: this project needs a llama.cpp "
                            "checkout with common/jinja (verified against 5ecbe1ac1)")
    endif()
endforeach()

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
    src/chat.cpp
    src/model.cpp
    src/npy_writer.cpp
    src/sampler.cpp
    src/session.cpp
    src/speculative.cpp
    src/thinking_budget.cpp
)
if(GGML_EXTRACTOR_MULTIMODAL)
    list(APPEND GGML_EXTRACTOR_SOURCES src/multimodal.cpp)
endif()
list(APPEND GGML_EXTRACTOR_SOURCES ${GGML_EXTRACTOR_JINJA_SOURCES})

add_library(ggml_extractor STATIC ${GGML_EXTRACTOR_SOURCES})
target_include_directories(ggml_extractor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Jinja headers ("jinja/lexer.h", "unicode.h") resolve inside the checkout.
# So does the staging header "llama-ext.h" (MTP support, WIP upstream).
target_include_directories(ggml_extractor PRIVATE
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/src"
    "${LLAMA_CPP_DIR}/vendor"
)
# Upstream sources build warning-free under their own flags, not ours.
set_source_files_properties(${GGML_EXTRACTOR_JINJA_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")
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
    workers_example  examples/06_workers.cpp
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
