#pragma once

#include <cstddef>
#include <string>

namespace ggml_extractor {

/// @brief Selects which token row to read from a captured ggml tensor.
///
/// ggml hidden-state tensors have shape `(n_embd, n_tokens, 1, 1)`.
/// The token axis (`ne[1]`) holds one row per decoded token. A selector
/// picks a single row:
///
/// - **Decode / prefill phase** (`llama_decode` with `n_tokens > 1`):
///   use ::last(), ::second_last(), or ::from_end(k) to address the last,
///   second-to-last, ..., k-th from last prompt token.
/// - **Generation / inference phase** (autoregressive loop with one new
///   token per `llama_decode`): use ::generated() to capture the row that
///   corresponds to the token just decoded. Technically this is also the
///   last row of the current decode batch, but the distinct factory makes
///   call-site intent explicit and keeps prefill vs. generate code paths
///   self-documenting.
///
/// Offsets are always counted **from the end** of the current decode batch,
/// never as absolute positions, so the same request works for prompts of any
/// length and for every generation step.
class TokenSelector {
public:
    /// @brief Selection mode.
    enum class Mode {
        /// Row `ne[1] - 1 - offset` of the current decode batch (prefill).
        kFromEnd,
        /// Row produced for the token just generated (inference loop).
        /// Encoded identically to offset 0; kept distinct for readability.
        kGenerated,
    };

    /// @brief Last token of the decode batch (`offset == 0`).
    static TokenSelector last() { return TokenSelector(Mode::kFromEnd, 0); }

    /// @brief Second-to-last token (`offset == 1`).
    static TokenSelector second_last() { return TokenSelector(Mode::kFromEnd, 1); }

    /// @brief Third-to-last token (`offset == 2`).
    static TokenSelector third_last() { return TokenSelector(Mode::kFromEnd, 2); }

    /// @brief Generic `offset`-from-end selector (`0` = last).
    static TokenSelector from_end(std::size_t offset) {
        return TokenSelector(Mode::kFromEnd, offset);
    }

    /// @brief Token produced by the current generation step.
    ///
    /// Use inside an autoregressive loop where each `llama_decode` feeds the
    /// newly sampled token. The extractor copies the last row of that decode,
    /// i.e. the hidden state of the current generated token.
    static TokenSelector generated() { return TokenSelector(Mode::kGenerated, 0); }

    /// @brief Selection mode.
    Mode mode() const { return mode_; }

    /// @brief Offset from the end (`0` = last row).
    std::size_t offset_from_end() const { return offset_; }

    /// @brief True for ::generated() selectors.
    bool is_generated() const { return mode_ == Mode::kGenerated; }

    /// @brief Human-readable form, e.g. `"last"`, `"from_end(2)"`, `"generated"`.
    std::string to_string() const {
        if (mode_ == Mode::kGenerated) {
            return "generated";
        }
        if (offset_ == 0) {
            return "last";
        }
        return "from_end(" + std::to_string(offset_) + ")";
    }

    bool operator==(const TokenSelector& other) const {
        return mode_ == other.mode_ && offset_ == other.offset_;
    }
    bool operator!=(const TokenSelector& other) const { return !(*this == other); }

private:
    TokenSelector(Mode mode, std::size_t offset) : mode_(mode), offset_(offset) {}

    Mode mode_ = Mode::kFromEnd;
    std::size_t offset_ = 0;
};

}  // namespace ggml_extractor
