// 02_batch.cpp — batch prefill: many prompts, one .npy.
//
// Each prompt is decoded independently (memory cleared between rows) and
// contributes one frame. Output shape: (n_prompts, n_requests, n_embd).
//
// Run:
//   ./build/batch_example -m model.gguf -o batch.npy -p "first" -p "second"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml_extractor/extractor.hpp"
#include "ggml_extractor/llama_session.hpp"
#include "ggml_extractor/token_selector.hpp"

int main(int argc, char** argv) {
    std::string model, output;
    std::vector<std::string> prompts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-m") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for -m");
            }
            model = argv[i];
        } else if (arg == "-o") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for -o");
            }
            output = argv[i];
        } else if (arg == "-p") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for -p");
            }
            prompts.emplace_back(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " -m <model.gguf> -o <out.npy> -p <prompt> [-p <prompt> ...]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (model.empty() || output.empty() || prompts.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " -m <model.gguf> -o <out.npy> -p <prompt> [-p ...]\n";
        return 1;
    }

    try {
        // Capture the last AND second-to-last token of two layers in one pass.
        // Same tensor, different token rows — supported by design.
        ggml_extractor::HiddenStateExtractor extractor({
            {"l_out-20", ggml_extractor::TokenSelector::last()},
            {"l_out-20", ggml_extractor::TokenSelector::second_last()},
            {"l_out-29", ggml_extractor::TokenSelector::last()},
        });
        ggml_extractor::LlamaSession session(model, extractor);

        for (std::size_t i = 0; i < prompts.size(); ++i) {
            try {
                session.decode(session.tokenize(prompts[i]));
                extractor.require_frame_complete();
                extractor.commit_frame();
            } catch (const std::exception& ex) {
                throw std::runtime_error("prompt " + std::to_string(i) + " failed: " +
                                         ex.what());
            }
            if ((i + 1) % 100 == 0 || i + 1 == prompts.size()) {
                std::cerr << "rows: " << (i + 1) << "/" << prompts.size() << "\n";
            }
        }

        extractor.save_npy(output);
        std::cout << "wrote " << output << " shape=(" << extractor.sequence_frame_count()
                  << ", " << extractor.request_count() << ", "
                  << extractor.common_embedding_width() << ")\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
