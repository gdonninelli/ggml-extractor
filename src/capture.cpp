#include "ggml_extractor/capture.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>

#include "ggml_extractor/npy_writer.hpp"

namespace ggml_extractor {
namespace {

void set_error(std::string& slot, const std::string& message) noexcept {
    try {
        if (slot.empty()) {
            slot = message;
        }
    } catch (...) {
        // The callback path must stay noexcept.
    }
}

}  // namespace

const float* HiddenStates::row(std::size_t frame, std::size_t request) const {
    if (frame >= n_frames || request >= n_requests) {
        throw std::runtime_error("hidden-state index out of range");
    }
    return data.data() + (frame * n_requests + request) * n_embd;
}

void HiddenStates::save_npy(const std::string& output_path) const {
    if (n_frames == 0) {
        throw std::runtime_error("no hidden states to save");
    }
    ::ggml_extractor::save_npy(output_path, data, {n_frames, n_requests, n_embd});
}

struct HiddenStateCapture::Impl {
    std::vector<ExtractionRequest> requests;
    /// Requested tensor name -> indices into `requests`. One tensor can serve
    /// several requests that pick different token rows. Transparent comparator
    /// so lookups take a string_view without allocating.
    std::map<std::string, std::vector<std::size_t>, std::less<>> by_name;

    std::vector<std::vector<float>> values;
    std::vector<bool> captured;
    std::vector<std::size_t> widths;
    std::string error;

    std::vector<float> sequence;
    std::size_t sequence_frames = 0;
    std::size_t sequence_embd = 0;

    explicit Impl(std::vector<ExtractionRequest> reqs) : requests(std::move(reqs)) {
        values.resize(requests.size());
        captured.resize(requests.size(), false);
        widths.resize(requests.size(), 0);
        for (std::size_t i = 0; i < requests.size(); ++i) {
            by_name[requests[i].tensor_name].push_back(i);
        }
    }

    std::size_t common_width() const {
        if (widths.empty() || widths[0] == 0) {
            return 0;
        }
        for (std::size_t i = 1; i < widths.size(); ++i) {
            if (widths[i] == 0) {
                return 0;
            }
            if (widths[i] != widths[0]) {
                throw std::runtime_error("inconsistent embedding widths across requests (" +
                                         std::to_string(widths[0]) + " vs " +
                                         std::to_string(widths[i]) + ")");
            }
        }
        return widths[0];
    }
};

HiddenStateCapture::HiddenStateCapture(std::vector<ExtractionRequest> requests) {
    if (requests.empty()) {
        throw std::runtime_error("HiddenStateCapture needs at least one request");
    }
    for (std::size_t i = 0; i < requests.size(); ++i) {
        if (requests[i].tensor_name.empty()) {
            throw std::runtime_error("request " + std::to_string(i) +
                                     " has an empty tensor name");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (requests[j].tensor_name == requests[i].tensor_name &&
                requests[j].token == requests[i].token) {
                throw std::runtime_error("duplicate request for tensor '" +
                                         requests[i].tensor_name + "' token " +
                                         requests[i].token.to_string());
            }
        }
    }
    impl_.reset(new Impl(std::move(requests)));
}

HiddenStateCapture::~HiddenStateCapture() = default;

bool HiddenStateCapture::filter(ggml_tensor* tensor, bool ask) noexcept {
    Impl& impl = *impl_;
    try {
        if (tensor == nullptr) {
            throw std::runtime_error("extractor callback received a null tensor");
        }
        const std::string_view name(tensor->name);

        const auto match = impl.by_name.find(name);
        if (match == impl.by_name.end()) {
            return false;
        }
        if (ask) {
            return true;
        }

        // Copy phase.
        if (tensor->type != GGML_TYPE_F32) {
            throw std::runtime_error("captured tensor is not float32: " + std::string(name));
        }
        if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("captured tensor has non-unit higher dims: " +
                                     std::string(name));
        }
        if (tensor->ne[0] <= 0 || tensor->ne[1] < 0) {
            throw std::runtime_error("captured tensor has empty dimensions: " +
                                     std::string(name));
        }
        if (tensor->ne[1] == 0) {
            // Not an error: llama.cpp gathers the final layer's output down to
            // the positions logits were requested for (`inp_out_ids`), so that
            // tensor is empty in any pass that produces no outputs. There is
            // nothing to copy here; another pass may still supply the row, and
            // require_frame_complete() reports it if none does.
            return false;
        }
        const auto embd64 = static_cast<std::uint64_t>(tensor->ne[0]);
        const auto rows64 = static_cast<std::uint64_t>(tensor->ne[1]);
        if (embd64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            rows64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("captured tensor dimensions overflow: " +
                                     std::string(name));
        }
        const auto embd = static_cast<std::size_t>(embd64);
        const auto n_rows = static_cast<std::size_t>(rows64);
        if (embd > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error("captured tensor row size overflows: " +
                                     std::string(name));
        }
        const std::size_t row_bytes = embd * sizeof(float);
        if (tensor->nb[0] != sizeof(float) || tensor->nb[1] != row_bytes) {
            throw std::runtime_error("captured tensor is not contiguous by token row: " +
                                     std::string(name));
        }

        bool copied_any = false;
        for (const std::size_t i : match->second) {
            const std::size_t offset = impl.requests[i].token.offset_from_end();
            if (offset >= n_rows) {
                // Two ways to land here: a prompt split across batches ends in
                // a short final batch, or this is the final layer, which
                // llama.cpp gathers down to the positions logits were asked
                // for. Report it rather than capture the wrong token.
                throw std::runtime_error(
                    "token " + impl.requests[i].token.to_string() +
                    " out of range for tensor '" + std::string(name) + "' with " +
                    std::to_string(n_rows) +
                    " row(s); the final layer only holds the rows you requested "
                    "logits for, and earlier layers only hold the current batch");
            }
            const std::size_t source_row = n_rows - 1 - offset;
            if (source_row != 0 &&
                tensor->nb[1] > std::numeric_limits<std::size_t>::max() / source_row) {
                throw std::runtime_error("captured tensor source offset overflows: " +
                                         std::string(name));
            }
            const std::size_t source_offset = source_row * tensor->nb[1];
            if (source_offset > std::numeric_limits<std::size_t>::max() - row_bytes ||
                source_offset + row_bytes > ggml_nbytes(tensor)) {
                throw std::runtime_error("captured tensor source range out of bounds: " +
                                         std::string(name));
            }

            if (impl.widths[i] == 0) {
                impl.widths[i] = embd;
                impl.values[i].resize(embd);
            } else if (impl.widths[i] != embd) {
                throw std::runtime_error("captured tensor width changed for '" +
                                         std::string(name) + "': expected " +
                                         std::to_string(impl.widths[i]) + ", got " +
                                         std::to_string(embd));
            } else if (impl.values[i].size() != embd) {
                impl.values[i].resize(embd);
            }

            ggml_backend_tensor_get(tensor, impl.values[i].data(), source_offset, row_bytes);
            impl.captured[i] = true;
            copied_any = true;
        }
        return copied_any;
    } catch (const std::exception& ex) {
        set_error(impl.error, ex.what());
        return false;
    } catch (...) {
        set_error(impl.error, "unknown extractor callback error");
        return false;
    }
}

void HiddenStateCapture::begin_frame() {
    Impl& impl = *impl_;
    std::fill(impl.captured.begin(), impl.captured.end(), false);
    impl.error.clear();
}

bool HiddenStateCapture::frame_complete() const {
    const Impl& impl = *impl_;
    if (!impl.error.empty()) {
        return false;
    }
    return std::all_of(impl.captured.begin(), impl.captured.end(), [](bool v) { return v; });
}

void HiddenStateCapture::require_frame_complete() const {
    const Impl& impl = *impl_;
    if (!impl.error.empty()) {
        throw std::runtime_error("extractor callback failed: " + impl.error);
    }
    for (std::size_t i = 0; i < impl.requests.size(); ++i) {
        if (!impl.captured[i]) {
            throw std::runtime_error(
                "missing hidden state for tensor '" + impl.requests[i].tensor_name +
                "' token " + impl.requests[i].token.to_string() +
                "; either that name is absent from this model's graph, or it is "
                "the final layer and the decode requested no logits (llama.cpp "
                "gathers the last layer down to the output positions, so "
                "extracting it needs logits_last = true)");
        }
    }
}

std::string HiddenStateCapture::error() const {
    return impl_->error;
}

void HiddenStateCapture::commit_frame() {
    Impl& impl = *impl_;
    require_frame_complete();

    const std::size_t width = impl.common_width();
    if (width == 0) {
        throw std::runtime_error("no hidden state captured yet");
    }
    if (impl.sequence_frames == 0) {
        impl.sequence_embd = width;
    } else if (impl.sequence_embd != width) {
        throw std::runtime_error("embedding width changed across frames");
    }

    const std::size_t base = impl.sequence.size();
    impl.sequence.resize(base + impl.requests.size() * width);
    for (std::size_t i = 0; i < impl.requests.size(); ++i) {
        const std::vector<float>& value = impl.values[i];
        if (value.size() != width) {
            throw std::runtime_error("inconsistent frame buffer for tensor '" +
                                     impl.requests[i].tensor_name + "'");
        }
        for (const float v : value) {
            if (!std::isfinite(v)) {
                throw std::runtime_error("captured hidden states contain non-finite values "
                                         "for tensor '" +
                                         impl.requests[i].tensor_name + "'");
            }
        }
        std::copy(value.begin(), value.end(), impl.sequence.begin() + base + i * width);
    }
    ++impl.sequence_frames;

    begin_frame();
}

std::size_t HiddenStateCapture::frame_count() const {
    return impl_->sequence_frames;
}

const std::vector<ExtractionRequest>& HiddenStateCapture::requests() const {
    return impl_->requests;
}

std::size_t HiddenStateCapture::embedding_width() const {
    return impl_->common_width();
}

HiddenStates HiddenStateCapture::take() {
    Impl& impl = *impl_;
    if (impl.sequence_frames == 0) {
        throw std::runtime_error("no committed frames to take (call commit_frame first)");
    }
    HiddenStates out;
    out.data = std::move(impl.sequence);
    out.n_frames = impl.sequence_frames;
    out.n_requests = impl.requests.size();
    out.n_embd = impl.sequence_embd;

    impl.sequence.clear();
    impl.sequence_frames = 0;
    impl.sequence_embd = 0;
    begin_frame();
    return out;
}

}  // namespace ggml_extractor
