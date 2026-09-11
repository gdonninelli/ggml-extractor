#include "ggml_extractor/backend.hpp"

#include "ggml-backend.h"
#include "llama.h"

#include <mutex>

namespace ggml_extractor {

void ensure_backend_initialized() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_backend_load_all();
        llama_backend_init();
    });
}

}  // namespace ggml_extractor
