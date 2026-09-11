#include "ggml_extractor/sampler.hpp"

#include "llama.h"

#include <stdexcept>

#include "ggml_extractor/session.hpp"

namespace ggml_extractor {

struct Sampler::Impl {
    llama_sampler* chain = nullptr;

    ~Impl() {
        if (chain != nullptr) {
            llama_sampler_free(chain);
        }
    }
};

Sampler::Sampler(const SamplerOptions& options) : impl_(new Impl()) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (options.temp <= 0.0f) {
        // Same as llama-server: non-positive temperature means greedy only.
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        // top_k <= 0 and top_p >= 1 each degrade to a no-op sampler.
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(options.top_k));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(options.top_p, /*min_keep=*/1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(options.temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(options.seed));
    }
    impl_->chain = chain;
}

Sampler::~Sampler() = default;

int32_t Sampler::sample(const Session& session) const {
    if (llama_get_logits_ith(session.handle(), -1) == nullptr) {
        throw std::runtime_error("no logits available (decode with logits_last = true)");
    }
    return llama_sampler_sample(impl_->chain, session.handle(), -1);
}

}  // namespace ggml_extractor
