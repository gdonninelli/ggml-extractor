#include "ggml_extractor/llama_session.hpp"

#include "ggml-backend.h"
#include "llama.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ggml_extractor {
namespace {

class BackendScope {
public:
    BackendScope() {
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~BackendScope() { llama_backend_free(); }
    BackendScope(const BackendScope&) = delete;
    BackendScope& operator=(const BackendScope&) = delete;
};

using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;

std::vector<int32_t> tokenize_prompt(const llama_vocab* vocab, std::string_view prompt) {
    if (prompt.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("prompt byte length exceeds llama.cpp limits");
    }
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    int32_t count = llama_tokenize(vocab, prompt.data(),
                                   static_cast<int32_t>(prompt.size()), nullptr, 0, add_bos,
                                   true);
    if (count >= 0) {
        throw std::runtime_error("prompt tokenization returned no required size");
    }
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("prompt token count exceeds llama.cpp limits");
    }
    std::vector<int32_t> tokens(static_cast<std::size_t>(-count));
    count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), add_bos,
                           true);
    if (count <= 0) {
        throw std::runtime_error("prompt tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

}  // namespace

struct LlamaSession::Impl {
    BackendScope backend;
    ModelPtr model{nullptr, &llama_model_free};
    ContextPtr context{nullptr, &llama_free};
    HiddenStateExtractor* extractor = nullptr;
    LlamaSessionOptions options;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_ctx_train = 0;
    uint32_t ctx_size = 0;
    int32_t n_past = 0;

    Impl(const std::string& model_path, HiddenStateExtractor& ext,
         LlamaSessionOptions opts)
        : extractor(&ext), options(std::move(opts)) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
        model_params.load_mtp = false;
        model.reset(llama_model_load_from_file(model_path.c_str(), model_params));
        if (!model) {
            throw std::runtime_error("failed to load model: " + model_path);
        }
        if (!options.required_architecture.empty()) {
            char arch[64] = {};
            const int32_t len = llama_model_meta_val_str(
                model.get(), "general.architecture", arch, sizeof(arch));
            if (len < 0 || static_cast<std::size_t>(len) >= sizeof(arch) ||
                std::string_view(arch) != options.required_architecture) {
                throw std::runtime_error("unexpected model architecture (required '" +
                                         options.required_architecture + "')");
            }
        }
        n_layer = llama_model_n_layer(model.get());
        n_embd = llama_model_n_embd(model.get());
        n_ctx_train = llama_model_n_ctx_train(model.get());
        if (n_layer <= 0 || n_embd <= 0) {
            throw std::runtime_error("model reports invalid hidden-state dimensions");
        }
    }

    void ensure_context(uint32_t needed) {
        if (needed == 0 || needed > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("requested context size is out of range");
        }
        if (context && needed <= ctx_size) {
            return;
        }
        uint32_t size = options.n_ctx != 0 ? options.n_ctx : needed;
        if (size < needed) {
            size = needed;
        }
        if (n_ctx_train > 0 && size > static_cast<uint32_t>(n_ctx_train)) {
            throw std::runtime_error("requested context exceeds model training context");
        }
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = size;
        ctx_params.n_batch = options.n_batch != 0 ? options.n_batch : size;
        extractor->attach(ctx_params);
        context.reset(llama_init_from_model(model.get(), ctx_params));
        if (!context) {
            throw std::runtime_error("failed to create llama.cpp context");
        }
        ctx_size = size;
        n_past = 0;
    }
};

LlamaSession::LlamaSession(const std::string& model_path,
                           HiddenStateExtractor& extractor,
                           const LlamaSessionOptions& options)
    : impl_(new Impl(model_path, extractor, options)) {}

LlamaSession::~LlamaSession() {
    delete impl_;
}

LlamaSession::LlamaSession(LlamaSession&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

LlamaSession& LlamaSession::operator=(LlamaSession&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

std::vector<int32_t> LlamaSession::tokenize(const std::string& prompt) const {
    if (prompt.empty()) {
        throw std::runtime_error("prompt must not be empty");
    }
    return tokenize_prompt(llama_model_get_vocab(impl_->model.get()), prompt);
}

void LlamaSession::decode(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error("cannot decode an empty token batch");
    }
    if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("prompt token count exceeds llama.cpp context limits");
    }
    impl_->ensure_context(static_cast<uint32_t>(tokens.size()));
    if (tokens.size() > impl_->ctx_size) {
        throw std::runtime_error("prompt token count exceeds the sized context");
    }

    // Prefill owns the KV cache: start from position 0.
    llama_memory_clear(llama_get_memory(impl_->context.get()), true);
    impl_->extractor->begin_frame();

    // llama_batch_get_one borrows the caller buffer; keep tokens alive.
    std::vector<int32_t> mutable_tokens = tokens;
    const int ret = llama_decode(
        impl_->context.get(),
        llama_batch_get_one(mutable_tokens.data(),
                            static_cast<int32_t>(mutable_tokens.size())));
    llama_synchronize(impl_->context.get());
    if (ret != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(ret));
    }
    impl_->n_past = static_cast<int32_t>(tokens.size());
}

void LlamaSession::decode_one(int32_t token) {
    if (!impl_->context) {
        // No prefill yet: size from options or from the training context.
        const uint32_t size =
            impl_->options.n_ctx != 0 ? impl_->options.n_ctx
                                      : static_cast<uint32_t>(impl_->n_ctx_train);
        impl_->ensure_context(size);
        llama_memory_clear(llama_get_memory(impl_->context.get()), true);
        impl_->n_past = 0;
    }
    if (impl_->n_past < 0 ||
        static_cast<uint32_t>(impl_->n_past) >= impl_->ctx_size) {
        throw std::runtime_error("generation exceeded the sized context");
    }
    impl_->extractor->begin_frame();

    int32_t mutable_token = token;
    llama_batch batch = llama_batch_get_one(&mutable_token, 1);
    batch.pos[0] = impl_->n_past;
    const int ret = llama_decode(impl_->context.get(), batch);
    llama_synchronize(impl_->context.get());
    if (ret != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(ret));
    }
    ++impl_->n_past;
}

int32_t LlamaSession::n_layer() const {
    return impl_->n_layer;
}

int32_t LlamaSession::n_embd() const {
    return impl_->n_embd;
}

int32_t LlamaSession::n_ctx_train() const {
    return impl_->n_ctx_train;
}

struct llama_context* LlamaSession::context_handle() {
    return impl_->context.get();
}

struct llama_model* LlamaSession::model_handle() const {
    return impl_->model.get();
}

}  // namespace ggml_extractor
