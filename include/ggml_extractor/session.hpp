#pragma once

#include <cstdint>
#include <memory>
#include <vector>

// Forward declaration keeps llama.h out of the public header surface.
struct llama_context;

namespace ggml_extractor {

class Model;
class HiddenStateCapture;

/// @brief Options for a Session. Sized once, at construction.
struct SessionOptions {
    /// Total context size in tokens, shared by `n_seq_max` sequences.
    /// Pick it for the worst case you intend to serve: the context is never
    /// resized, which is what lets the KV cache survive.
    uint32_t n_ctx = 4096;
    /// Logical batch size — the most tokens one `llama_decode` may carry.
    /// `0` = same as `n_ctx`.
    uint32_t n_batch = 0;
    /// Physical batch size. `0` = llama.cpp default.
    ///
    /// Note for `from_end(k)` requests with `k > 0`: a prompt longer than
    /// `n_ubatch` is evaluated in several passes, and the last one may hold
    /// fewer than `k + 1` tokens — in which case the row does not exist and
    /// the capture reports it. Raise `n_ubatch` past your prompt length when
    /// you need offsets other than the last token.
    uint32_t n_ubatch = 0;
    /// Independent sequences (conversation slots) the context can hold.
    uint32_t n_seq_max = 1;
    /// Generation threads. `0` = llama.cpp default.
    int32_t n_threads = 0;
    /// Prompt-processing threads. `0` = llama.cpp default.
    int32_t n_threads_batch = 0;
};

/// @brief One inference stream over a shared Model: owns a `llama_context`
/// and the permanently installed ggml eval callback.
///
/// @par Why the callback is permanent
/// llama.cpp copies `cb_eval` into the context at creation and offers no
/// setter, so extraction cannot be wired in later. The Session therefore
/// installs its own trampoline once, up front, and switches extraction on and
/// off behind it with arm(). While nothing is armed the trampoline answers
/// "no" for every graph node, which lets the ggml scheduler coalesce each
/// graph split back into a single submission — so ordinary inference runs at
/// full speed on the same context that extraction uses.
///
/// @par Lifetime
/// The context is created in the constructor and never recreated, so the KV
/// cache lives as long as the Session. Use reset() to free one sequence.
///
/// @code
/// auto model = Model::load("gemma4.gguf");
/// SessionOptions options;
/// options.n_ctx = 8192;
/// Session session(model, options);
///
/// // plain inference — nothing armed, full speed
/// session.decode(0, model->tokenize("hello"));
/// int32_t next = session.sample_greedy();
///
/// // extraction — same context, same KV cache
/// HiddenStateCapture capture({{"l_out-20", TokenSelector::last()}});
/// {
///     auto armed = session.arm(capture);
///     session.decode_one(0, next);
///     capture.commit_frame();
/// }
/// HiddenStates states = capture.take();
/// @endcode
///
/// @note Not thread-safe: one Session per worker thread. Sessions over the
/// same Model are independent and may run concurrently.
class Session {
public:
    /// @brief Create the context. Keeps the Model alive for its own lifetime.
    /// @throws std::runtime_error when the context cannot be created.
    Session(std::shared_ptr<Model> model, const SessionOptions& options = SessionOptions{});

    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = delete;
    Session& operator=(Session&&) = delete;

    /// @brief Scope guard returned by arm(): extraction is live while it
    /// exists, and stops when it goes out of scope (including on exception).
    class Armed {
    public:
        ~Armed();
        Armed(Armed&& other) noexcept;
        Armed(const Armed&) = delete;
        Armed& operator=(const Armed&) = delete;
        Armed& operator=(Armed&&) = delete;

    private:
        friend class Session;
        explicit Armed(Session* session);
        Session* session_;
    };

    /// @brief Route the evaluated graph through `capture` until the returned
    /// guard dies. Discards the capture's in-progress frame; committed frames
    /// are kept.
    /// @throws std::runtime_error when a capture is already armed.
    [[nodiscard]] Armed arm(HiddenStateCapture& capture);

    /// @brief True while a capture is armed.
    bool armed() const;

    /// @brief Decode a batch of tokens into sequence `seq`, continuing from
    /// that sequence's current position. Splits across `n_batch` as needed.
    /// @param logits_last Compute logits for the final token — needed only
    /// when you intend to sample next.
    /// @throws std::runtime_error when the sequence would exceed the context.
    void decode(int32_t seq, const std::vector<int32_t>& tokens, bool logits_last = true);

    /// @brief Decode one token into `seq` (a generation step), with logits.
    void decode_one(int32_t seq, int32_t token);

    /// @brief Next position for `seq`, i.e. how many tokens it holds.
    int32_t n_past(int32_t seq) const;

    /// @brief Drop everything cached for `seq`, freeing the slot for reuse.
    void reset(int32_t seq);

    /// @brief Drop every sequence.
    void reset_all();

    /// @brief Greedy argmax over the most recently decoded logits.
    /// @throws std::runtime_error when no logits are available.
    int32_t sample_greedy() const;

    /// @brief The shared Model.
    Model& model() const;

    /// @brief Raw context handle for sampling or any llama.cpp call this
    /// wrapper does not cover. The Session retains ownership.
    llama_context* handle() const;

    /// @brief Context size as actually allocated by llama.cpp (it rounds up).
    uint32_t n_ctx() const;
    /// @brief Per-sequence context size.
    uint32_t n_ctx_seq() const;
    /// @brief Logical batch size.
    uint32_t n_batch() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
