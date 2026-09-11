#include "ggml_extractor/chat.hpp"

#include "llama.h"

#include <stdexcept>

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include "ggml_extractor/model.hpp"

namespace ggml_extractor {

struct ChatFormat::Impl {
    std::string source;
    jinja::program prog;
    std::string bos;
    std::string eos;
    bool vocab_adds_bos = false;
};

ChatFormat::ChatFormat(const Model& model, const std::string& template_override)
    : impl_(new Impl()) {
    const char* source = template_override.empty()
                             ? llama_model_chat_template(model.handle(), nullptr)
                             : template_override.c_str();
    if (source == nullptr || *source == '\0') {
        throw std::runtime_error(
            "model has no chat template in its metadata; pass an explicit template_override");
    }
    impl_->source = source;

    jinja::lexer lexer;
    impl_->prog = jinja::parse_from_tokens(lexer.tokenize(impl_->source));

    const int32_t bos = llama_vocab_bos(model.vocab());
    if (bos >= 0) {
        impl_->bos = model.token_to_piece(bos);
    }
    const int32_t eos = llama_vocab_eos(model.vocab());
    if (eos >= 0) {
        impl_->eos = model.token_to_piece(eos);
    }
    impl_->vocab_adds_bos = llama_vocab_get_add_bos(model.vocab());
}

ChatFormat::~ChatFormat() = default;

std::string ChatFormat::render(const std::vector<ChatMessage>& messages,
                               const ChatOptions& options) const {
    auto rendered = jinja::mk_val<jinja::value_array>();
    for (const ChatMessage& message : messages) {
        auto item = jinja::mk_val<jinja::value_object>();
        item->insert("role", jinja::mk_val<jinja::value_string>(message.role));
        item->insert("content", jinja::mk_val<jinja::value_string>(message.content));
        rendered->push_back(item);
    }

    jinja::context ctx(impl_->source);
    ctx.set_val("messages", rendered);
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(impl_->bos));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(impl_->eos));
    ctx.set_val("add_generation_prompt",
                jinja::mk_val<jinja::value_bool>(options.add_generation_prompt));
    ctx.set_val("enable_thinking", jinja::mk_val<jinja::value_bool>(options.enable_thinking));

    jinja::runtime runtime(ctx);
    const jinja::value result = runtime.execute(impl_->prog);
    std::string prompt = jinja::runtime::gather_string_parts(result)->as_string().str();

    // Model::tokenize adds BOS when the vocabulary wants it, so a BOS the
    // template already emitted would appear twice.
    if (impl_->vocab_adds_bos && !impl_->bos.empty() && prompt.rfind(impl_->bos, 0) == 0) {
        prompt.erase(0, impl_->bos.size());
    }
    return prompt;
}

const std::string& ChatFormat::source() const {
    return impl_->source;
}

}  // namespace ggml_extractor
