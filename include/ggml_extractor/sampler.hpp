#pragma once

#include <cstdint>
#include <memory>

// Forward declarations keep llama.h out of the public header surface.
struct llama_sampler;

namespace ggml_extractor {

class Session;

/// @brief Word-picking settings. Defaults match llama.cpp's own defaults.
struct SamplerOptions {
    /// Keep only the K most likely tokens. `<= 0` = disabled.
    int32_t top_k = 40;
    /// Keep tokens up to this cumulative probability. `>= 1` = disabled.
    float top_p = 0.95f;
    /// Randomness. `<= 0` = always pick the most likely token (greedy).
    float temp = 0.8f;
    /// Random seed. `0xFFFFFFFF` = random (matches LLAMA_DEFAULT_SEED).
    uint32_t seed = 0xFFFFFFFFu;
};

/// @brief Picks the next token from a Session's last logits.
///
/// Chain order is llama.cpp's convention: top-k, top-p, temperature, draw.
/// A Session holds no picker state, so one Sampler serves a whole worker
/// thread — but like Session, it is not safe to share across threads.
class Sampler {
public:
    explicit Sampler(const SamplerOptions& options = SamplerOptions{});

    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) = delete;
    Sampler& operator=(Sampler&&) = delete;

    /// @brief Draw one token from the Session's most recent logits.
    /// The Session must have decoded with logits enabled (decode_one always
    /// does; decode() does unless logits_last = false was passed).
    /// @throws std::runtime_error when no logits are available.
    int32_t sample(const Session& session) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ggml_extractor
