#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml_extractor/extractor.hpp"

// Forward declarations to keep llama.h out of the public header surface.
struct llama_model;
struct llama_context;

namespace ggml_extractor {

/// @brief Options for LlamaSession (thin convenience wrapper over llama.cpp).
struct LlamaSessionOptions {
    /// Passed to `llama_model_default_params().n_gpu_layers`. `-1` = auto.
    int32_t n_gpu_layers = -1;
    /// Optional context size override. `0` = size each decode from the token
    /// count (simple, matches the reference tools). Set explicitly when you
    /// need KV-cache reuse across generation steps.
    uint32_t n_ctx = 0;
    /// Passed through to `llama_context_default_params().n_batch`.
    /// `0` = same as the sized context.
    uint32_t n_batch = 0;
    /// When non-empty, require this `general.architecture` metadata value
    /// (e.g. `"gemma4"`). Empty = accept any architecture.
    std::string required_architecture;
};

/// @brief Convenience owner of `llama_model` + `llama_context` wired to a
/// HiddenStateExtractor.
///
/// The session does **not** own the extractor: construct the extractor first
/// (it defines *what* to capture), then hand it to the session (which defines
/// *where* inference runs). The extractor must outlive the session.
///
/// @code
/// HiddenStateExtractor extractor({{"l_out-20", TokenSelector::last()}});
/// LlamaSession session("model.gguf", extractor);
///
/// auto tokens = session.tokenize("hello world");
/// session.decode(tokens);                 // prefill
/// extractor.require_frame_complete();
/// extractor.commit_frame();
/// extractor.save_npy("out.npy");
/// @endcode
///
/// For generation loops, call decode_one() per sampled token while the
/// extractor holds TokenSelector::generated() requests.
class LlamaSession {
public:
    /// @brief Load the model and keep (but do not yet create) the context.
    /// @param model_path Path to a `.gguf` file.
    /// @param extractor Extractor to attach (must outlive this session).
    /// @param options GPU layers / context sizing / architecture guard.
    /// @throws std::runtime_error when the model cannot be loaded.
    LlamaSession(const std::string& model_path,
                 HiddenStateExtractor& extractor,
                 const LlamaSessionOptions& options = LlamaSessionOptions{});

    ~LlamaSession();
    LlamaSession(const LlamaSession&) = delete;
    LlamaSession& operator=(const LlamaSession&) = delete;
    LlamaSession(LlamaSession&&) noexcept;
    LlamaSession& operator=(LlamaSession&&) noexcept;

    /// @brief Tokenize a prompt (adds BOS per model metadata, parses special).
    std::vector<int32_t> tokenize(const std::string& prompt) const;

    /// @brief Decode a full prompt batch (prefill). Sizes/creates the context
    /// on first call, clears KV memory, runs begin_frame/decode/synchronize.
    /// After return, query the extractor (require_frame_complete/commit).
    void decode(const std::vector<int32_t>& tokens);

    /// @brief Decode exactly one token (generation step).
    /// Creates the context on first call if needed; otherwise reuses it.
    /// Clears nothing except per-step state — KV cache is preserved by
    /// llama.cpp across calls sharing the same context.
    void decode_one(int32_t token);

    /// @brief Number of transformer layers reported by the model.
    int32_t n_layer() const;
    /// @brief Hidden width reported by the model.
    int32_t n_embd() const;
    /// @brief Training context length reported by the model.
    int32_t n_ctx_train() const;

    /// @brief Raw handles for advanced use (sampling, logits, BOS/EOS).
    /// Valid after construction (model) / first decode (context).
    /// The session retains ownership; do not free.
    struct llama_context* context_handle();
    struct llama_model* model_handle() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace ggml_extractor
