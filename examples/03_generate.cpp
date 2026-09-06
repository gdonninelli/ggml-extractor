// 03_generate.cpp — autoregressive generation: capture the CURRENT generated
// token at every step (greedy argmax over logits, no sampler dependency).
//
//   1. prefill the prompt,
//   2. greedy-sample the next token from the last logits,
//   3. decode_one(token) per step — the extractor copies the GENERATED row,
//   4. commit one frame per generated token.
//
// Output shape: (n_generated, n_requests, n_embd).
//
// Run:
//   ./build/generate_example -m model.gguf -p "The capital of Italy is" \
//       -o gen.npy -n 32

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml_extractor/extractor.hpp"
#include "ggml_extractor/llama_session.hpp"
#include "ggml_extractor/token_selector.hpp"
#include "llama.h"

namespace {

int32_t greedy_next(llama_context* ctx, const llama_vocab* vocab) {
    float* logits = llama_get_logits(ctx);
    if (logits == nullptr) {
        throw std::runtime_error("llama_get_logits returned null (enable logits?)");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("invalid vocabulary size");
    }
    int32_t best = 0;
    float best_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, prompt, output;
    int n_predict = 32;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* opt) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + opt);
            }
            return argv[i];
        };
        if (arg == "-m") {
            model = need_value("-m");
        } else if (arg == "-p") {
            prompt = need_value("-p");
        } else if (arg == "-o") {
            output = need_value("-o");
        } else if (arg == "-n") {
            n_predict = std::stoi(need_value("-n"));
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " -m <model.gguf> -p <prompt> -o <out.npy> [-n <tokens>]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (model.empty() || prompt.empty() || output.empty() || n_predict <= 0) {
        std::cerr << "Usage: " << argv[0]
                  << " -m <model.gguf> -p <prompt> -o <out.npy> [-n <tokens>]\n";
        return 1;
    }

    try {
        ggml_extractor::HiddenStateExtractor extractor({
            {"l_out-20", ggml_extractor::TokenSelector::generated()},
            {"l_out-29", ggml_extractor::TokenSelector::generated()},
        });

        ggml_extractor::LlamaSessionOptions opts;
        opts.n_ctx = 4096;  // fixed context so the KV cache survives across steps
        ggml_extractor::LlamaSession session(model, extractor, opts);

        // Prefill. Its frame is intentionally dropped: we only accumulate
        // GENERATED frames below. Position tracking inside the session
        // continues from the prompt length.
        session.decode(session.tokenize(prompt));
        extractor.require_frame_complete();
        extractor.begin_frame();  // discard prefill capture, keep KV + position

        const llama_vocab* vocab = llama_model_get_vocab(session.model_handle());
        const llama_token eos = llama_vocab_eos(vocab);

        std::vector<int32_t> generated;
        generated.reserve(static_cast<std::size_t>(n_predict));
        for (int step = 0; step < n_predict; ++step) {
            const int32_t next = greedy_next(session.context_handle(), vocab);
            generated.push_back(next);

            session.decode_one(next);  // captures the GENERATED row
            extractor.require_frame_complete();
            extractor.commit_frame();  // one frame per generated token

            if (next == eos) {
                break;
            }
        }

        extractor.save_npy(output);
        std::cout << "generated " << generated.size() << " token(s); wrote " << output
                  << " shape=(" << extractor.sequence_frame_count() << ", "
                  << extractor.request_count() << ", "
                  << extractor.common_embedding_width() << ")\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
