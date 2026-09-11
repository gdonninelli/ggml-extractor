#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml_extractor {

class Model;
class Session;

/// @brief Options for the multimodal encoder.
struct MultimodalOptions {
    /// Run the vision/audio encoder on the GPU when one is available.
    bool use_gpu = true;
    /// Encoder threads.
    int n_threads = 4;
    /// Let libmtmd print its own encode timings.
    bool print_timings = false;
};

/// @brief Vision and audio front end (llama.cpp's `libmtmd`) over a shared
/// Model.
///
/// This owns only the projector (`mmproj`) weights — the text weights come
/// from the Model — so adding image support to an already-loaded model costs
/// one extra file, not a second copy of the model.
///
/// eval() runs `llama_decode` on the Session's context, which means an armed
/// HiddenStateCapture observes those decodes exactly as it observes a text
/// prefill: hidden states for an image prompt come out of the same code path.
///
/// @code
/// auto model = Model::load("gemma4.gguf");
/// Session session(model, SessionOptions{});
/// Multimodal vision(model, "mmproj-gemma4.gguf");
///
/// vision.eval(session, 0, "<__media__>\nWhat is in this picture?", {"cat.png"});
/// int32_t next = session.sample_greedy();
/// @endcode
///
/// @note Not thread-safe, and neither is the Session it decodes into.
class Multimodal {
public:
    /// @brief Load the projector and bind it to `model`'s text weights.
    /// @param mmproj_path Path to the `mmproj-*.gguf` for this model.
    /// @throws std::runtime_error when the projector cannot be loaded.
    Multimodal(std::shared_ptr<Model> model, const std::string& mmproj_path,
               const MultimodalOptions& options = MultimodalOptions{});

    ~Multimodal();
    Multimodal(const Multimodal&) = delete;
    Multimodal& operator=(const Multimodal&) = delete;

    /// @brief Whether this projector accepts images.
    bool supports_vision() const;
    /// @brief Whether this projector accepts audio.
    bool supports_audio() const;

    /// @brief The placeholder to put in a prompt where media should be
    /// substituted. llama.cpp's default is `"<__media__>"`.
    std::string marker() const;

    /// @brief Encode `media_paths` and decode prompt + media into `seq`,
    /// continuing from that sequence's current position.
    ///
    /// `prompt` must contain exactly one marker() per media path; the markers
    /// are replaced by the encoded media in order. Image and audio files are
    /// detected by content (anything stb_image or miniaudio can read).
    ///
    /// BOS is added only when `seq` is empty, so follow-up turns in a
    /// conversation do not pick up a second one.
    ///
    /// @param logits_last Compute logits for the final token — needed only
    /// when you intend to sample next.
    /// @throws std::runtime_error on a marker/media count mismatch, an
    /// unreadable file, or a decode failure.
    void eval(Session& session, int32_t seq, const std::string& prompt,
              const std::vector<std::string>& media_paths, bool logits_last = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
