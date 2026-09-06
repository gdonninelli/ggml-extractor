// 01_prefill.cpp — minimal prefill extraction: one prompt, N tensors, last token.
//
// Build (after configuring with -DLLAMA_CPP_DIR=<path to llama.cpp>):
//   cmake --build build --target prefill_example
// Run:
//   ./build/prefill_example -m model.gguf -p "hello world" -o out.npy
//
// Output shape: (n_requests, n_embd) — one row per ExtractionRequest.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml_extractor/extractor.hpp"
#include "ggml_extractor/llama_session.hpp"
#include "ggml_extractor/token_selector.hpp"

namespace {

void usage(const char* prog) {
    std::cout << "Usage: " << prog << " -m <model.gguf> -p <prompt> -o <out.npy>\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, prompt, output;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& opt) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + opt);
            }
            return argv[i];
        };
        if (arg == "-m") {
            model = need_value(arg);
        } else if (arg == "-p") {
            prompt = need_value(arg);
        } else if (arg == "-o") {
            output = need_value(arg);
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (model.empty() || prompt.empty() || output.empty()) {
        usage(argv[0]);
        return 1;
    }

    try {
        // 1. Declare WHAT to capture: exact ggml tensor names + token rows.
        ggml_extractor::HiddenStateExtractor extractor({
            {"inp_scaled", ggml_extractor::TokenSelector::last()},
            {"l_out-20", ggml_extractor::TokenSelector::last()},
            {"l_out-25", ggml_extractor::TokenSelector::last()},
        });

        // 2. Open the model; the session attaches the extractor to llama.cpp.
        ggml_extractor::LlamaSession session(model, extractor);

        // 3. Prefill decode. The extractor copies each requested row.
        session.decode(session.tokenize(prompt));
        extractor.require_frame_complete();
        extractor.commit_frame();

        // 4. Persist: shape (3, n_embd).
        extractor.save_npy(output);
        std::cout << "wrote " << output << " shape=(3, " << extractor.common_embedding_width()
                  << ")\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
