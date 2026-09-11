#include "ggml_extractor/model.hpp"

#include "llama.h"

#include <limits>
#include <stdexcept>
#include <string_view>

#include "ggml_extractor/backend.hpp"

namespace ggml_extractor {

struct Model::Impl {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_ctx_train = 0;
    int32_t n_vocab = 0;
    std::string architecture;

    ~Impl() {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

namespace {

std::string read_metadata(llama_model* model, const char* key) {
    char buffer[128] = {};
    const int32_t len = llama_model_meta_val_str(model, key, buffer, sizeof(buffer));
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buffer)) {
        return {};
    }
    return std::string(buffer, static_cast<std::size_t>(len));
}

}  // namespace

Model::Model(const std::string& model_path, const ModelOptions& options)
    : impl_(new Impl()) {
    ensure_backend_initialized();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;

    impl_->model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (impl_->model == nullptr) {
        throw std::runtime_error("failed to load model: " + model_path);
    }

    impl_->architecture = read_metadata(impl_->model, "general.architecture");
    if (!options.required_architecture.empty() &&
        impl_->architecture != options.required_architecture) {
        throw std::runtime_error("unexpected model architecture '" + impl_->architecture +
                                 "' (required '" + options.required_architecture + "')");
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);
    impl_->n_layer = llama_model_n_layer(impl_->model);
    impl_->n_embd = llama_model_n_embd(impl_->model);
    impl_->n_ctx_train = llama_model_n_ctx_train(impl_->model);
    impl_->n_vocab = llama_vocab_n_tokens(impl_->vocab);
    if (impl_->n_layer <= 0 || impl_->n_embd <= 0) {
        throw std::runtime_error("model reports invalid hidden-state dimensions");
    }
}

std::shared_ptr<Model> Model::load(const std::string& model_path,
                                   const ModelOptions& options) {
    return std::shared_ptr<Model>(new Model(model_path, options));
}

Model::~Model() = default;

std::vector<int32_t> Model::tokenize(const std::string& text, bool add_special,
                                     bool parse_special) const {
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("text byte length exceeds llama.cpp limits");
    }
    const int32_t text_len = static_cast<int32_t>(text.size());

    int32_t count = llama_tokenize(impl_->vocab, text.data(), text_len, nullptr, 0,
                                   add_special, parse_special);
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("token count exceeds llama.cpp limits");
    }
    if (count > 0) {
        throw std::runtime_error("tokenization did not report a required size");
    }
    if (count == 0) {
        return {};
    }

    std::vector<int32_t> tokens(static_cast<std::size_t>(-count));
    count = llama_tokenize(impl_->vocab, text.data(), text_len, tokens.data(),
                           static_cast<int32_t>(tokens.size()), add_special,
                           parse_special);
    if (count < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

std::string Model::token_to_piece(int32_t token) const {
    char buffer[256];
    int32_t len = llama_token_to_piece(impl_->vocab, token, buffer, sizeof(buffer), 0,
                                       /*special=*/true);
    if (len >= 0) {
        return std::string(buffer, static_cast<std::size_t>(len));
    }
    // Negative length is the required buffer size.
    std::string piece(static_cast<std::size_t>(-len), '\0');
    len = llama_token_to_piece(impl_->vocab, token, piece.data(),
                               static_cast<int32_t>(piece.size()), 0, /*special=*/true);
    if (len < 0) {
        throw std::runtime_error("failed to render token " + std::to_string(token));
    }
    piece.resize(static_cast<std::size_t>(len));
    return piece;
}

bool Model::is_eog(int32_t token) const {
    return llama_vocab_is_eog(impl_->vocab, token);
}

int32_t Model::n_layer() const {
    return impl_->n_layer;
}

int32_t Model::n_embd() const {
    return impl_->n_embd;
}

int32_t Model::n_ctx_train() const {
    return impl_->n_ctx_train;
}

int32_t Model::n_vocab() const {
    return impl_->n_vocab;
}

std::string Model::architecture() const {
    return impl_->architecture;
}

llama_model* Model::handle() const {
    return impl_->model;
}

const llama_vocab* Model::vocab() const {
    return impl_->vocab;
}

}  // namespace ggml_extractor
