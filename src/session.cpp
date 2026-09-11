#include "ggml_extractor/session.hpp"

#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "ggml_extractor/capture.hpp"
#include "ggml_extractor/model.hpp"

namespace ggml_extractor {

struct Session::Impl {
    std::shared_ptr<Model> model;
    llama_context* context = nullptr;
    llama_batch batch{};
    bool batch_owned = false;
    HiddenStateCapture* capture = nullptr;
    uint32_t n_ctx = 0;
    uint32_t n_ctx_seq = 0;
    uint32_t n_batch = 0;
    uint32_t n_seq_max = 1;

    ~Impl() {
        if (batch_owned) {
            llama_batch_free(batch);
        }
        if (context != nullptr) {
            llama_free(context);
        }
    }

    /// Permanently installed as the context's `cb_eval`. With no capture
    /// armed this answers "no" for every node, which lets the ggml scheduler
    /// merge the whole graph split into one submission.
    static bool trampoline(ggml_tensor* tensor, bool ask, void* user_data) noexcept {
        auto* self = static_cast<Impl*>(user_data);
        if (self == nullptr || self->capture == nullptr) {
            return false;
        }
        return self->capture->filter(tensor, ask);
    }

    void check_seq(int32_t seq) const {
        if (seq < 0 || static_cast<uint32_t>(seq) >= n_seq_max) {
            throw std::runtime_error("sequence id " + std::to_string(seq) +
                                     " is out of range (n_seq_max = " +
                                     std::to_string(n_seq_max) + ")");
        }
    }

    llama_memory_t memory() const { return llama_get_memory(context); }

    int32_t n_past(int32_t seq) const {
        return llama_memory_seq_pos_max(memory(), seq) + 1;
    }

    void decode_tokens(int32_t seq, const int32_t* tokens, std::size_t count,
                       bool logits_last) {
        check_seq(seq);
        if (count == 0) {
            throw std::runtime_error("cannot decode an empty token batch");
        }
        const int32_t start = n_past(seq);
        if (static_cast<uint64_t>(start) + count > n_ctx_seq) {
            throw std::runtime_error(
                "sequence " + std::to_string(seq) + " would exceed the context (" +
                std::to_string(start) + " + " + std::to_string(count) + " > " +
                std::to_string(n_ctx_seq) + "); raise SessionOptions::n_ctx or reset it");
        }

        for (std::size_t offset = 0; offset < count; offset += n_batch) {
            const std::size_t chunk = std::min<std::size_t>(n_batch, count - offset);
            batch.n_tokens = static_cast<int32_t>(chunk);
            for (std::size_t i = 0; i < chunk; ++i) {
                batch.token[i] = tokens[offset + i];
                batch.pos[i] = start + static_cast<int32_t>(offset + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = 0;
            }
            const bool is_last_chunk = offset + chunk == count;
            if (logits_last && is_last_chunk) {
                batch.logits[chunk - 1] = 1;
            }
            const int ret = llama_decode(context, batch);
            if (ret != 0) {
                throw std::runtime_error("llama_decode failed with code " +
                                         std::to_string(ret));
            }
        }
        llama_synchronize(context);
    }
};

Session::Session(std::shared_ptr<Model> model, const SessionOptions& options)
    : impl_(new Impl()) {
    if (!model) {
        throw std::runtime_error("Session requires a loaded Model");
    }
    if (options.n_ctx == 0) {
        throw std::runtime_error("SessionOptions::n_ctx must be non-zero");
    }
    if (options.n_seq_max == 0) {
        throw std::runtime_error("SessionOptions::n_seq_max must be non-zero");
    }
    impl_->model = std::move(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = options.n_ctx;
    ctx_params.n_batch = options.n_batch != 0 ? options.n_batch : options.n_ctx;
    if (options.n_ubatch != 0) {
        ctx_params.n_ubatch = options.n_ubatch;
    }
    ctx_params.n_seq_max = options.n_seq_max;
    if (options.n_threads != 0) {
        ctx_params.n_threads = options.n_threads;
    }
    if (options.n_threads_batch != 0) {
        ctx_params.n_threads_batch = options.n_threads_batch;
    }
    // Installed once and never changed: arm() switches the behaviour behind it.
    ctx_params.cb_eval = &Impl::trampoline;
    ctx_params.cb_eval_user_data = impl_.get();

    impl_->context = llama_init_from_model(impl_->model->handle(), ctx_params);
    if (impl_->context == nullptr) {
        throw std::runtime_error("failed to create llama.cpp context");
    }

    impl_->n_ctx = llama_n_ctx(impl_->context);
    impl_->n_ctx_seq = llama_n_ctx_seq(impl_->context);
    impl_->n_batch = llama_n_batch(impl_->context);
    impl_->n_seq_max = options.n_seq_max;
    if (impl_->n_batch == 0) {
        throw std::runtime_error("llama.cpp reported a zero batch size");
    }

    impl_->batch = llama_batch_init(static_cast<int32_t>(impl_->n_batch), 0, 1);
    impl_->batch_owned = true;
}

Session::~Session() = default;

Session::Armed::Armed(Session* session) : session_(session) {}

Session::Armed::Armed(Armed&& other) noexcept : session_(other.session_) {
    other.session_ = nullptr;
}

Session::Armed::~Armed() {
    if (session_ != nullptr) {
        session_->impl_->capture = nullptr;
    }
}

Session::Armed Session::arm(HiddenStateCapture& capture) {
    if (impl_->capture != nullptr) {
        throw std::runtime_error("a capture is already armed on this session");
    }
    capture.begin_frame();
    impl_->capture = &capture;
    return Armed(this);
}

bool Session::armed() const {
    return impl_->capture != nullptr;
}

void Session::decode(int32_t seq, const std::vector<int32_t>& tokens, bool logits_last) {
    impl_->decode_tokens(seq, tokens.data(), tokens.size(), logits_last);
}

void Session::decode_one(int32_t seq, int32_t token) {
    impl_->decode_tokens(seq, &token, 1, /*logits_last=*/true);
}

int32_t Session::n_past(int32_t seq) const {
    impl_->check_seq(seq);
    return impl_->n_past(seq);
}

void Session::reset(int32_t seq) {
    impl_->check_seq(seq);
    llama_memory_seq_rm(impl_->memory(), seq, -1, -1);
}

void Session::reset_all() {
    llama_memory_clear(impl_->memory(), true);
}

int32_t Session::sample_greedy() const {
    const float* logits = llama_get_logits_ith(impl_->context, -1);
    if (logits == nullptr) {
        throw std::runtime_error("no logits available (decode with logits_last = true)");
    }
    const int32_t n_vocab = impl_->model->n_vocab();
    if (n_vocab <= 0) {
        throw std::runtime_error("invalid vocabulary size");
    }
    int32_t best = 0;
    float best_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    return best;
}

Model& Session::model() const {
    return *impl_->model;
}

llama_context* Session::handle() const {
    return impl_->context;
}

uint32_t Session::n_ctx() const {
    return impl_->n_ctx;
}

uint32_t Session::n_ctx_seq() const {
    return impl_->n_ctx_seq;
}

uint32_t Session::n_seq_max() const {
    return impl_->n_seq_max;
}

uint32_t Session::n_batch() const {
    return impl_->n_batch;
}

}  // namespace ggml_extractor
