#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declarations keep llama.h out of the public header surface.
struct llama_model;
struct llama_vocab;

namespace ggml_extractor {

/// @brief Options for loading a Model.
struct ModelOptions {
    /// Layers to offload to the GPU. `-1` = as many as fit.
    int32_t n_gpu_layers = -1;
    /// When non-empty, require this `general.architecture` metadata value
    /// (e.g. `"gemma4"`). Empty accepts any architecture.
    std::string required_architecture;
};

/// @brief Model weights, loaded once and shared by every Session.
///
/// This is the expensive object: loading it reads gigabytes from disk and
/// fills GPU memory. In a long-running service you load it at startup and
/// keep it for the process lifetime, then create cheap Sessions against it.
///
/// A `llama_model` is read-only during inference, so one Model can back any
/// number of concurrent Sessions.
///
/// @code
/// auto model = Model::load("gemma4.gguf");
/// Session a(model);   // independent KV caches, safe to run in parallel
/// Session b(model);
/// @endcode
class Model {
public:
    /// @brief Load a GGUF file. Initialises the ggml backends if needed.
    /// @throws std::runtime_error when the file cannot be loaded or the
    /// architecture guard fails.
    static std::shared_ptr<Model> load(const std::string& model_path,
                                       const ModelOptions& options = ModelOptions{});

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// @brief Tokenize text. `add_special` adds BOS/EOS per model metadata,
    /// `parse_special` recognises control tokens written out in the text.
    std::vector<int32_t> tokenize(const std::string& text,
                                  bool add_special = true,
                                  bool parse_special = true) const;

    /// @brief Render one token back to text (special tokens included).
    std::string token_to_piece(int32_t token) const;

    /// @brief True for end-of-generation tokens (EOS and friends).
    bool is_eog(int32_t token) const;

    /// @brief Transformer layer count.
    int32_t n_layer() const;
    /// @brief Hidden width.
    int32_t n_embd() const;
    /// @brief Training context length.
    int32_t n_ctx_train() const;
    /// @brief Vocabulary size.
    int32_t n_vocab() const;
    /// @brief `general.architecture` metadata value, e.g. `"gemma4"`.
    std::string architecture() const;

    /// @brief Raw handles. The Model retains ownership; do not free.
    llama_model* handle() const;
    const llama_vocab* vocab() const;

private:
    Model(const std::string& model_path, const ModelOptions& options);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
