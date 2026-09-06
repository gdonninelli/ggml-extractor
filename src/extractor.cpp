#include "ggml_extractor/extractor.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "ggml_extractor/npy_writer.hpp"

namespace ggml_extractor {
namespace {

void set_error(std::string& slot, const char* message) noexcept {
    try {
        slot = message;
    } catch (...) {
        // Callback path must stay noexcept.
    }
}

}  // namespace

struct HiddenStateExtractor::Impl {
    std::vector<ExtractionRequest> requests;
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
    }
};

HiddenStateExtractor::HiddenStateExtractor(std::vector<ExtractionRequest> requests)
    : impl_(nullptr) {
    if (requests.empty()) {
        throw std::runtime_error("HiddenStateExtractor needs at least one request");
    }
    for (std::size_t i = 0; i < requests.size(); ++i) {
        if (requests[i].tensor_name.empty()) {
            throw std::runtime_error("request " + std::to_string(i) + " has an empty tensor name");
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
    impl_ = new Impl(std::move(requests));
}

HiddenStateExtractor::~HiddenStateExtractor() {
    delete impl_;
}

HiddenStateExtractor::HiddenStateExtractor(HiddenStateExtractor&& other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

HiddenStateExtractor& HiddenStateExtractor::operator=(HiddenStateExtractor&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void HiddenStateExtractor::attach(llama_context_params& params) {
    params.cb_eval = &HiddenStateExtractor::callback;
    params.cb_eval_user_data = this;
}

bool HiddenStateExtractor::callback(ggml_tensor* tensor, bool ask, void* user_data) noexcept {
    auto* self = static_cast<HiddenStateExtractor*>(user_data);
    if (self == nullptr) {
        return false;
    }
    return self->filter(tensor, ask);
}

bool HiddenStateExtractor::filter(ggml_tensor* tensor, bool ask) noexcept {
    Impl& impl = *impl_;
    try {
        if (tensor == nullptr) {
            throw std::runtime_error("extractor callback received a null tensor");
        }
        const std::string_view name(tensor->name);

        bool name_matches = false;
        for (const auto& req : impl.requests) {
            if (name == req.tensor_name) {
                name_matches = true;
                break;
            }
        }
        if (!name_matches) {
            return false;
        }
        if (ask) {
            return true;
        }

        // Copy phase: one tensor may serve several requests that share the
        // name but select different token rows (e.g. last + second-last).
        if (tensor->type != GGML_TYPE_F32) {
            throw std::runtime_error("captured tensor is not float32: " + std::string(name));
        }
        if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("captured tensor has non-unit higher dims: " +
                                     std::string(name));
        }
        if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
            throw std::runtime_error("captured tensor has empty dimensions: " +
                                     std::string(name));
        }
        const auto embd64 = static_cast<std::uint64_t>(tensor->ne[0]);
        const auto rows64 = static_cast<std::uint64_t>(tensor->ne[1]);
        if (embd64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            rows64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("captured tensor dimensions overflow: " +
                                     std::string(name));
        }
        const std::size_t embd = static_cast<std::size_t>(embd64);
        const std::size_t n_rows = static_cast<std::size_t>(rows64);
        if (embd == 0 || embd > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error("captured tensor row size overflows: " +
                                     std::string(name));
        }
        const std::size_t row_bytes = embd * sizeof(float);
        if (tensor->nb[0] != sizeof(float) || tensor->nb[1] != row_bytes) {
            throw std::runtime_error("captured tensor is not contiguous by token row: " +
                                     std::string(name));
        }

        bool copied_any = false;
        for (std::size_t i = 0; i < impl.requests.size(); ++i) {
            if (name != impl.requests[i].tensor_name) {
                continue;
            }
            const std::size_t offset = impl.requests[i].token.offset_from_end();
            if (offset >= n_rows) {
                throw std::runtime_error("token " + impl.requests[i].token.to_string() +
                                         " out of range for tensor '" + std::string(name) +
                                         "' with " + std::to_string(n_rows) + " row(s)");
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

void HiddenStateExtractor::begin_frame() {
    Impl& impl = *impl_;
    std::fill(impl.captured.begin(), impl.captured.end(), false);
    impl.error.clear();
}

bool HiddenStateExtractor::frame_complete() const {
    const Impl& impl = *impl_;
    if (!impl.error.empty()) {
        return false;
    }
    return std::all_of(impl.captured.begin(), impl.captured.end(),
                       [](bool v) { return v; });
}

void HiddenStateExtractor::require_frame_complete() const {
    const Impl& impl = *impl_;
    if (!impl.error.empty()) {
        throw std::runtime_error("extractor callback failed: " + impl.error);
    }
    for (std::size_t i = 0; i < impl.requests.size(); ++i) {
        if (!impl.captured[i]) {
            throw std::runtime_error("missing hidden state for tensor '" +
                                     impl.requests[i].tensor_name + "' token " +
                                     impl.requests[i].token.to_string());
        }
    }
}

std::string HiddenStateExtractor::error() const {
    return impl_->error;
}

std::size_t HiddenStateExtractor::request_count() const {
    return impl_->requests.size();
}

const ExtractionRequest& HiddenStateExtractor::request(std::size_t index) const {
    if (index >= impl_->requests.size()) {
        throw std::runtime_error("request index out of range");
    }
    return impl_->requests[index];
}

const std::vector<ExtractionRequest>& HiddenStateExtractor::requests() const {
    return impl_->requests;
}

std::size_t HiddenStateExtractor::embedding_width(std::size_t index) const {
    if (index >= impl_->widths.size()) {
        throw std::runtime_error("request index out of range");
    }
    return impl_->widths[index];
}

std::size_t HiddenStateExtractor::common_embedding_width() const {
    const Impl& impl = *impl_;
    if (impl.widths.empty() || impl.widths[0] == 0) {
        throw std::runtime_error("no hidden state captured yet");
    }
    const std::size_t width = impl.widths[0];
    for (std::size_t i = 1; i < impl.widths.size(); ++i) {
        if (impl.widths[i] == 0) {
            throw std::runtime_error("request '" + impl.requests[i].tensor_name +
                                     "' has no captured width yet");
        }
        if (impl.widths[i] != width) {
            throw std::runtime_error("inconsistent embedding widths across requests (" +
                                     std::to_string(width) + " vs " +
                                     std::to_string(impl.widths[i]) +
                                     "); save_npy needs uniform widths");
        }
    }
    return width;
}

const std::vector<float>& HiddenStateExtractor::frame_row(std::size_t index) const {
    const Impl& impl = *impl_;
    if (index >= impl.requests.size()) {
        throw std::runtime_error("request index out of range");
    }
    if (!impl.captured[index]) {
        throw std::runtime_error("no captured row for tensor '" +
                                 impl.requests[index].tensor_name + "' token " +
                                 impl.requests[index].token.to_string());
    }
    return impl.values[index];
}

std::vector<float> HiddenStateExtractor::frame_flat() const {
    const Impl& impl = *impl_;
    require_frame_complete();
    const std::size_t width = common_embedding_width();
    std::vector<float> flat;
    flat.reserve(impl.requests.size() * width);
    for (std::size_t i = 0; i < impl.requests.size(); ++i) {
        if (impl.values[i].size() != width) {
            throw std::runtime_error("inconsistent frame buffer for tensor '" +
                                     impl.requests[i].tensor_name + "'");
        }
        flat.insert(flat.end(), impl.values[i].begin(), impl.values[i].end());
    }
    return flat;
}

void HiddenStateExtractor::commit_frame() {
    Impl& impl = *impl_;
    require_frame_complete();
    const std::vector<float> flat = frame_flat();
    for (float v : flat) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("captured hidden states contain non-finite values");
        }
    }
    const std::size_t width = common_embedding_width();
    if (impl.sequence_frames == 0) {
        impl.sequence_embd = width;
    } else if (impl.sequence_embd != width) {
        throw std::runtime_error("embedding width changed across frames");
    }
    impl.sequence.insert(impl.sequence.end(), flat.begin(), flat.end());
    ++impl.sequence_frames;
}

std::size_t HiddenStateExtractor::sequence_frame_count() const {
    return impl_->sequence_frames;
}

bool HiddenStateExtractor::has_sequence() const {
    return impl_->sequence_frames > 0;
}

void HiddenStateExtractor::clear_sequence() {
    Impl& impl = *impl_;
    impl.sequence.clear();
    impl.sequence_frames = 0;
    impl.sequence_embd = 0;
}

void HiddenStateExtractor::save_npy(const std::string& output_path) const {
    const Impl& impl = *impl_;
    if (impl.sequence_frames == 0) {
        throw std::runtime_error("no committed frames to save (call commit_frame first)");
    }
    ::ggml_extractor::save_npy(output_path, impl.sequence,
                               {impl.sequence_frames, impl.requests.size(),
                                impl.sequence_embd});
}

void HiddenStateExtractor::save_current_frame_npy(const std::string& output_path) const {
    const std::vector<float> flat = frame_flat();
    const std::size_t width = common_embedding_width();
    ::ggml_extractor::save_npy(output_path, flat, {impl_->requests.size(), width});
}

}  // namespace ggml_extractor
