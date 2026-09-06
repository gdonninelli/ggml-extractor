#pragma once

#include <string>
#include <utility>

#include "ggml_extractor/token_selector.hpp"

namespace ggml_extractor {

/// @brief One (tensor, token) pair to capture.
///
/// @par Tensor name
/// `tensor_name` must **exactly** equal the ggml tensor name
/// (`tensor->name`) as it appears in the GGUF / ggml graph. Examples from
/// llama.cpp instrumented builds:
/// - `"inp_scaled"` — scaled input embeddings (row 0 in full captures).
/// - `"l_out-<layer>"` — transformer block output, e.g. `"l_out-20"`.
/// Any other exact tensor name present in the evaluated graph works as well.
/// Matching is case-sensitive exact equality; non-matching tensors are
/// ignored by the `ask` phase of the callback.
///
/// @par Token
/// `token` selects which row of `ne[1]` to copy (see TokenSelector).
/// Each request carries its own selector, so one extractor can capture e.g.
/// the last token of layer 20 and the second-to-last token of layer 25 in a
/// single decode.
struct ExtractionRequest {
    /// Exact ggml tensor name, e.g. `"l_out-20"` or `"inp_scaled"`.
    std::string tensor_name;
    /// Which token row to copy from that tensor.
    TokenSelector token = TokenSelector::last();

    ExtractionRequest() = default;
    ExtractionRequest(std::string name, TokenSelector selector)
        : tensor_name(std::move(name)), token(selector) {}
};

}  // namespace ggml_extractor
