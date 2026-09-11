#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ggml_extractor/extraction_request.hpp"

// Forward declaration so this header does not require ggml.h.
struct ggml_tensor;

namespace ggml_extractor {

/// @brief Captured hidden states, float32, C-order
/// `(n_frames, n_requests, n_embd)`.
///
/// Returned in memory by HiddenStateCapture::take(). Writing a `.npy` file is
/// optional — a service normally serialises `data` straight onto the wire.
struct HiddenStates {
    /// Flat buffer of `n_frames * n_requests * n_embd` floats.
    std::vector<float> data;
    /// One frame per committed decode step.
    std::size_t n_frames = 0;
    /// Requests, in the order they were given to the capture.
    std::size_t n_requests = 0;
    /// Hidden width, uniform across requests and frames.
    std::size_t n_embd = 0;

    bool empty() const { return n_frames == 0; }

    /// @brief Pointer to the `n_embd` floats for one (frame, request) pair.
    /// @throws std::runtime_error when either index is out of range.
    const float* row(std::size_t frame, std::size_t request) const;

    /// @brief Write as a NumPy `.npy` file. Refuses to overwrite an existing
    /// path; publishes atomically.
    void save_npy(const std::string& output_path) const;
};

/// @brief One extraction job: which rows to capture, and the buffers they
/// land in.
///
/// A capture is deliberately short-lived and owns no llama.cpp state.
/// Construct one per request, arm it on a Session for the decodes it should
/// observe, take() the result, and drop it — a long-running service can cycle
/// through thousands of captures against a single Session, each asking for
/// different tensors.
///
/// @par Frames
/// A frame is one row of the output: the requested tensors as of some point
/// in time. Several `llama_decode` calls may contribute to one frame (a long
/// prompt split across batches, or an interleaved image + text prompt), with
/// each decode overwriting the last — so a frame committed after a whole
/// prefill holds the final prompt token. commit_frame() closes the current
/// frame and opens the next, which makes a generation loop one commit per
/// sampled token.
///
/// @note filter() is `noexcept`: failures are recorded and re-thrown by
/// require_frame_complete() on the decoding thread.
/// @note Not thread-safe. One capture per inference stream.
class HiddenStateCapture {
public:
    /// @brief Build a capture for the given (tensor, token) requests.
    /// @param requests Non-empty; tensor names must be non-empty and
    /// (tensor, token) pairs must be unique.
    /// @throws std::runtime_error on empty or invalid requests.
    explicit HiddenStateCapture(std::vector<ExtractionRequest> requests);

    ~HiddenStateCapture();
    HiddenStateCapture(const HiddenStateCapture&) = delete;
    HiddenStateCapture& operator=(const HiddenStateCapture&) = delete;

    /// @brief The ggml eval-callback body: `ask` = "do you want this tensor?",
    /// then `!ask` = "copy it now".
    ///
    /// Session installs this for you. Call it directly only when driving a
    /// raw ggml graph loop with no llama_context.
    bool filter(ggml_tensor* tensor, bool ask) noexcept;

    /// @brief Discard the in-progress frame and start a fresh one.
    /// Session::arm() does this; call it manually to drop a capture you no
    /// longer want (for example a prefill frame before a generation loop).
    void begin_frame();

    /// @brief True when every request has been copied for the current frame.
    bool frame_complete() const;

    /// @brief Throw with tensor and token details when the current frame is
    /// incomplete or filter() recorded an error.
    void require_frame_complete() const;

    /// @brief Last filter() error (empty when healthy).
    std::string error() const;

    /// @brief Validate the current frame, append it, and begin the next one.
    /// @throws std::runtime_error when the frame is incomplete, holds
    /// non-finite values, or its width disagrees with earlier frames.
    void commit_frame();

    /// @brief Committed frames so far.
    std::size_t frame_count() const;

    /// @brief The requests, in construction order.
    const std::vector<ExtractionRequest>& requests() const;

    /// @brief Hidden width observed so far (`0` before the first capture).
    /// @throws std::runtime_error when requests disagree on the width.
    std::size_t embedding_width() const;

    /// @brief Move the committed frames out, leaving the capture empty and
    /// ready for reuse.
    /// @throws std::runtime_error when no frame has been committed.
    HiddenStates take();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
