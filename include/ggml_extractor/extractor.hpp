#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ggml_extractor/extraction_request.hpp"

// Forward declarations so this header does not require llama.h / ggml.h
// at include time. The .cpp file includes them.
struct ggml_tensor;
struct llama_context_params;

namespace ggml_extractor {

/// @brief Object-oriented hidden-state extractor for ggml / llama.cpp.
///
/// The user instantiates **one** object and passes the list of tensors +
/// token positions to capture. The object then plugs into any llama.cpp
/// inference context and copies the requested rows out of the evaluated
/// ggml graph.
///
/// @par Typical prefill (decode) usage
/// @code
/// HiddenStateExtractor extractor({
///     {"inp_scaled", TokenSelector::last()},
///     {"l_out-20",   TokenSelector::last()},
///     {"l_out-25",   TokenSelector::second_last()},
/// });
/// llama_context_params ctx_params = llama_context_default_params();
/// extractor.attach(ctx_params);          // sets cb_eval + user data
/// // ... create llama_context from ctx_params ...
/// extractor.begin_frame();               // before each llama_decode
/// llama_decode(ctx, batch);
/// llama_synchronize(ctx);
/// extractor.require_frame_complete();    // throws with details if missing
/// extractor.commit_frame();              // append to sequence buffer
/// extractor.save_npy("hidden_states.npy");
/// @endcode
///
/// @par Typical generation (inference) usage
/// @code
/// HiddenStateExtractor extractor({
///     {"l_out-20", TokenSelector::generated()},
///     {"l_out-29", TokenSelector::generated()},
/// });
/// // prefill prompt first, then per generated token:
/// for (llama_token tok : generated_tokens) {
///     extractor.begin_frame();
///     llama_decode(ctx, llama_batch_get_one(&tok, 1));
///     llama_synchronize(ctx);
///     extractor.require_frame_complete();
///     extractor.commit_frame();  // one frame per generated token
/// }
/// // saved shape: (n_generated, n_requests, n_embd)
/// extractor.save_npy("generated_states.npy");
/// @endcode
///
/// @par Raw ggml usage (no llama.cpp)
/// Call HiddenStateExtractor::callback() (or the Filter() helper) from any
/// loop that visits every ggml_tensor* of the evaluated graph, with `ask`
/// first and then the copy phase — the same two-phase protocol llama.cpp
/// uses for `cb_eval`.
///
/// @note The callback is `noexcept`: failures are recorded in error() and
/// re-thrown by require_frame_complete() on the calling thread.
/// @note Not thread-safe for concurrent decodes; use one extractor per
/// inference stream.
class HiddenStateExtractor {
public:
    /// @brief Build an extractor for the given (tensor, token) requests.
    /// @param requests Non-empty list; tensor names must be non-empty.
    /// Duplicates (same tensor + same token) are rejected.
    /// @throws std::runtime_error on empty / invalid requests.
    explicit HiddenStateExtractor(std::vector<ExtractionRequest> requests);

    ~HiddenStateExtractor();
    HiddenStateExtractor(const HiddenStateExtractor&) = delete;
    HiddenStateExtractor& operator=(const HiddenStateExtractor&) = delete;
    HiddenStateExtractor(HiddenStateExtractor&&) noexcept;
    HiddenStateExtractor& operator=(HiddenStateExtractor&&) noexcept;

    // ------------------------------------------------------------------
    // Plug-and-play wiring
    // ------------------------------------------------------------------

    /// @brief Attach to llama.cpp: sets `params.cb_eval` and
    /// `params.cb_eval_user_data` to this extractor.
    /// Overwrites any previously set callback.
    void attach(llama_context_params& params);

    /// @brief Static trampoline suitable for `llama_context_params.cb_eval`.
    /// @param tensor Tensor visited by the evaluator (may be nullptr on error).
    /// @param ask Two-phase protocol: `true` = "should I keep this tensor?",
    /// `false` = "copy it now".
    /// @param user_data Pointer to a HiddenStateExtractor (set by attach()).
    /// @return true if the tensor was requested (ask) or copied (eval).
    static bool callback(ggml_tensor* tensor, bool ask, void* user_data) noexcept;

    /// @brief Direct filter for raw ggml loops (same semantics as callback
    /// but bound to this instance).
    bool filter(ggml_tensor* tensor, bool ask) noexcept;

    // ------------------------------------------------------------------
    // Per-frame lifecycle (one frame == one llama_decode)
    // ------------------------------------------------------------------

    /// @brief Reset captured flags, scratch buffers and the error string.
    /// Call before every `llama_decode`.
    void begin_frame();

    /// @brief True when every request has been copied for the current frame.
    bool frame_complete() const;

    /// @brief Throw `std::runtime_error` if the frame is incomplete or the
    /// callback recorded an error. Call after `llama_synchronize`.
    void require_frame_complete() const;

    /// @brief Last callback error (empty when healthy).
    std::string error() const;

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// @brief Number of extraction requests (constructor order is kept).
    std::size_t request_count() const;

    /// @brief The i-th request (0-based, constructor order).
    const ExtractionRequest& request(std::size_t index) const;

    /// @brief All requests (constructor order).
    const std::vector<ExtractionRequest>& requests() const;

    /// @brief Embedding width observed for request `index` in the current
    /// frame (`0` before the first successful capture).
    std::size_t embedding_width(std::size_t index) const;

    /// @brief Common embedding width across requests.
    /// @throws std::runtime_error if no frame captured yet or widths differ.
    std::size_t common_embedding_width() const;

    /// @brief Read-only view of the current frame for request `index`.
    /// @throws std::runtime_error if that request was not captured.
    const std::vector<float>& frame_row(std::size_t index) const;

    /// @brief Current frame flattened in request order:
    /// `[req0(embd) | req1(embd) | ...]`.
    /// @throws std::runtime_error if the frame is incomplete or widths are
    /// inconsistent.
    std::vector<float> frame_flat() const;

    // ------------------------------------------------------------------
    // Multi-frame accumulation + .npy output
    // ------------------------------------------------------------------

    /// @brief Validate the current frame (finite + complete) and append a
    /// copy to the internal sequence buffer.
    /// Call once per decode (prefill contributes 1 frame; a generation loop
    /// contributes 1 frame per generated token).
    void commit_frame();

    /// @brief Number of committed frames in the sequence buffer.
    std::size_t sequence_frame_count() const;

    /// @brief True when at least one frame has been committed.
    bool has_sequence() const;

    /// @brief Drop all committed frames (keeps the requests).
    void clear_sequence();

    /// @brief Save committed frames to a NumPy `.npy` file.
    /// Shape is `(n_frames, n_requests, n_embd)`.
    /// @param output_path Destination (must not already exist; atomic publish).
    /// @throws std::runtime_error when empty, widths differ, or I/O fails.
    void save_npy(const std::string& output_path) const;

    /// @brief Save only the current (uncommitted) frame.
    /// Shape is `(n_requests, n_embd)`.
    void save_current_frame_npy(const std::string& output_path) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace ggml_extractor
