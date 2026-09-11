#pragma once

namespace ggml_extractor {

/// @brief Initialise the ggml backend registry and llama.cpp, exactly once
/// per process.
///
/// Safe to call from any thread and any number of times; every call after the
/// first is a no-op. Model::load() calls it for you.
///
/// There is deliberately no matching teardown. llama.cpp's backend registry is
/// process-global state shared by every model and context, so releasing it
/// from an object destructor would pull it out from under objects that are
/// still alive. The registry is left for process exit to reclaim.
void ensure_backend_initialized();

}  // namespace ggml_extractor
