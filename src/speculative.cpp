#include "ggml_extractor/speculative.hpp"

#include "llama.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

// Staging header: MTP support (nextn outputs, shared-memory check). Present
// in the pinned llama.cpp checkout; not part of the stable API.
#include "llama-ext.h"

#include "ggml_extractor/model.hpp"
#include "ggml_extractor/session.hpp"

namespace ggml_extractor {

namespace {

// Non-owning batch view: memory comes from llama_batch_init, like upstream.
struct RoundBatch {
    llama_batch batch{};
    bool token_owned = false;

    RoundBatch(int32_t n_tokens_cap, int32_t n_embd, int32_t n_seq_max) {
        batch = llama_batch_init(n_tokens_cap, n_embd, n_seq_max);
        if (n_embd > 0) {
            batch.token =
                static_cast<llama_token*>(std::malloc(sizeof(llama_token) * n_tokens_cap));
            token_owned = batch.token != nullptr;
        }
    }

    ~RoundBatch() {
        if (token_owned) {
            std::free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    RoundBatch(const RoundBatch&) = delete;
    RoundBatch& operator=(const RoundBatch&) = delete;
};

int32_t argmax(const float* logits, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

}  // namespace

struct Speculative::Impl {
    Session* session = nullptr;  // target; must outlive this
    llama_model* draft_model = nullptr;
    llama_context* draft_ctx = nullptr;
    RoundBatch* draft_batch = nullptr;   // 1 token + 1 hidden row
    RoundBatch* verify_batch = nullptr;  // id_last + drafts, tokens only
    int n_drafts = 3;
    uint32_t n_seq_max = 1;
    int32_t n_embd_out = 0;
    int32_t n_vocab = 0;
    int n_mtp_layers = 0;
    int64_t n_drafted = 0;
    int64_t n_matched = 0;
    std::vector<float> pending;  // draft input row: hidden state at pos - 1

    ~Impl() {
        delete verify_batch;
        delete draft_batch;
        if (draft_ctx != nullptr) {
            llama_free(draft_ctx);
        }
        if (draft_model != nullptr) {
            llama_model_free(draft_model);
        }
    }

    llama_memory_t target_memory() const { return llama_get_memory(session->handle()); }

    void check_decode(int code, const char* what) const {
        if (code != 0) {
            throw std::runtime_error(std::string(what) + " failed with code " +
                                     std::to_string(code));
        }
    }

    // Hidden state row for output `idx` of the target's most recent decode.
    // NOTE: the target runs unmasked, so rows are indexed by position in
    // the decode (0-based) — negative indices are invalid there.
    void read_pending(int32_t idx) {
        const float* row = llama_get_embeddings_nextn_ith(session->handle(), idx);
        if (row == nullptr) {
            throw std::runtime_error(
                "no draft priming data (generate after a decode with logits)");
        }
        std::memcpy(pending.data(), row, pending.size() * sizeof(float));
    }
};

Speculative::Speculative(Session& session, const std::string& draft_path,
                         const SpeculativeOptions& options)
    : impl_(new Impl()) {
    if (options.n_drafts <= 0) {
        throw std::runtime_error("SpeculativeOptions::n_drafts must be positive");
    }
    if (static_cast<uint32_t>(options.n_drafts) + 1 > session.n_batch()) {
        throw std::runtime_error(
            "a round verifies n_drafts + 1 tokens at once, which exceeds the session "
            "batch size; raise n_batch or lower n_drafts");
    }
    impl_->session = &session;
    impl_->n_drafts = options.n_drafts;
    impl_->n_seq_max = session.n_seq_max();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    impl_->draft_model = llama_model_load_from_file(draft_path.c_str(), model_params);
    if (impl_->draft_model == nullptr) {
        throw std::runtime_error("failed to load draft model: " + draft_path);
    }
    impl_->n_mtp_layers = llama_model_n_layer_nextn(impl_->draft_model);
    if (impl_->n_mtp_layers <= 0) {
        throw std::runtime_error("not an MTP draft head: " + draft_path);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = session.n_ctx();
    ctx_params.n_batch = 32;
    ctx_params.n_seq_max = session.n_seq_max();
    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.ctx_other = session.handle();  // share the session's memory
    ctx_params.n_rs_seq = 0;
    impl_->draft_ctx = llama_init_from_model(impl_->draft_model, ctx_params);
    if (impl_->draft_ctx == nullptr) {
        throw std::runtime_error("failed to create the draft context");
    }
    if (llama_get_ctx_other(impl_->draft_ctx) != session.handle()) {
        throw std::runtime_error(
            "draft head does not share this session's memory; only Gemma-style "
            "shared-memory MTP heads are supported");
    }

    impl_->n_embd_out = llama_model_n_embd_out(impl_->draft_model);
    if (impl_->n_embd_out != llama_model_n_embd_out(session.model().handle()) ||
        impl_->n_embd_out <= 0) {
        throw std::runtime_error("draft and target hidden widths differ");
    }
    impl_->n_vocab = session.model().n_vocab();
    if (llama_vocab_n_tokens(llama_model_get_vocab(impl_->draft_model)) != impl_->n_vocab) {
        throw std::runtime_error("draft and target vocabularies differ");
    }
    impl_->pending.assign(static_cast<std::size_t>(impl_->n_embd_out), 0.0f);

    // Draft input rows come from these outputs from here on.
    llama_set_embeddings_nextn(session.handle(), true, /*masked=*/false);
    llama_set_embeddings_nextn(impl_->draft_ctx, true, /*masked=*/true);

    impl_->draft_batch = new RoundBatch(1, impl_->n_embd_out, session.n_seq_max());
    impl_->verify_batch =
        new RoundBatch(options.n_drafts + 1, 0, session.n_seq_max());
}

Speculative::~Speculative() = default;

int Speculative::generate(int32_t seq, int n_predict, std::string& out) {
    Impl& impl = *impl_;
    if (n_predict <= 0) {
        throw std::runtime_error("n_predict must be positive");
    }
    if (seq < 0 || static_cast<uint32_t>(seq) >= impl.n_seq_max) {
        throw std::runtime_error("sequence id out of range for this session");
    }
    if (impl.session->armed()) {
        throw std::runtime_error(
            "cannot run the draft while a capture is armed; record with "
            "sequential generation instead");
    }
    Model& model = impl.session->model();
    llama_context* target = impl.session->handle();
    const int32_t n_vocab = impl.n_vocab;
    const std::size_t row_bytes =
        static_cast<std::size_t>(impl.n_embd_out) * sizeof(float);

    int produced = 0;
    int32_t pos = impl.session->n_past(seq);

    // The round's fresh token: sampled but never decoded. Every verify
    // batch starts with it at a fresh position, so no token is ever
    // decoded twice and every decode starts where memory left off.
    int32_t fresh = -1;

    auto emit = [&](int32_t token) { out += model.token_to_piece(token); };
    auto target_logits = [&](int32_t idx) {
        const float* logits = llama_get_logits_ith(target, idx);
        if (logits == nullptr) {
            throw std::runtime_error("target logits missing during verification");
        }
        return logits;
    };

    // One plain step first: it emits the first token and primes the draft
    // (hidden state of a known single-row decode).
    {
        const float* logits = llama_get_logits_ith(target, -1);
        if (logits == nullptr) {
            throw std::runtime_error("no logits available (decode with logits_last = true)");
        }
        const int32_t first = argmax(logits, n_vocab);
        emit(first);
        ++produced;
        if (model.is_eog(first) || produced >= n_predict) {
            return produced;
        }
        impl.session->decode_one(seq, first);
        ++pos;
        const float* first_logits = llama_get_logits_ith(target, 0);
        if (first_logits == nullptr) {
            throw std::runtime_error("target logits missing after first decode");
        }
        fresh = argmax(first_logits, n_vocab);
        impl.read_pending(0);  // decode_one carries exactly one token
    }

    std::vector<int32_t> drafts;
    drafts.reserve(static_cast<std::size_t>(impl.n_drafts));
    llama_batch& draft = impl.draft_batch->batch;
    llama_batch& verify = impl.verify_batch->batch;

    // The round's head token was emitted as last round's fresh token — except
    // in the first round, where it is new. Emitting it again would duplicate
    // every round boundary.
    bool head_is_new = true;

    while (produced < n_predict) {
        // --- draft: propose n tokens for pos + 1 and on ---
        // First input pairs the fresh token with the pending hidden state
        // (hidden at pos - 1, token at pos); later steps reuse the position
        // and carry the hidden state forward explicitly.
        drafts.clear();
        draft.n_tokens = 1;
        draft.token[0] = fresh;
        draft.pos[0] = pos;
        draft.n_seq_id[0] = 1;
        draft.seq_id[0][0] = seq;
        draft.logits[0] = 1;
        std::memcpy(draft.embd, impl.pending.data(), row_bytes);
        for (int step = 0; step < impl.n_drafts; ++step) {
            impl.check_decode(llama_decode(impl.draft_ctx, draft), "draft decode");
            llama_synchronize(impl.draft_ctx);
            const float* dlogits = llama_get_logits_ith(impl.draft_ctx, -1);
            if (dlogits == nullptr) {
                throw std::runtime_error("draft logits missing");
            }
            const int32_t next = argmax(dlogits, n_vocab);
            drafts.push_back(next);
            const float* h = llama_get_embeddings_nextn_ith(impl.draft_ctx, 0);
            if (h == nullptr) {
                throw std::runtime_error("draft hidden state missing");
            }
            draft.token[0] = next;  // same position, new hidden row
            std::memcpy(draft.embd, h, row_bytes);
        }
        const int32_t n_new = static_cast<int32_t>(drafts.size());
        impl.n_drafted += n_new;

        // --- verify: fresh token plus drafts, one row of logits each ---
        verify.n_tokens = n_new + 1;
        verify.token[0] = fresh;
        verify.pos[0] = pos;
        verify.n_seq_id[0] = 1;
        verify.seq_id[0][0] = seq;
        verify.logits[0] = 1;
        for (int32_t i = 0; i < n_new; ++i) {
            verify.token[i + 1] = drafts[static_cast<std::size_t>(i)];
            verify.pos[i + 1] = pos + 1 + i;
            verify.n_seq_id[i + 1] = 1;
            verify.seq_id[i + 1][0] = seq;
            verify.logits[i + 1] = 1;
        }
        impl.check_decode(llama_decode(target, verify), "verify decode");
        llama_synchronize(target);

        // --- accept: longest matching prefix, plus one fresh token ---
        // drafts[i] is tested against row i, which predicts its position.
        // The incoming fresh token needs no test: sampled, not drafted.
        const int32_t head = fresh;
        int32_t matched = 0;
        while (matched < n_new) {
            const int32_t pick = argmax(target_logits(matched), n_vocab);
            if (pick != drafts[static_cast<std::size_t>(matched)]) {
                break;
            }
            ++matched;
        }
        // Row `matched` predicted the first rejected position (or one past
        // the drafts when all matched): its pick is the next fresh token.
        fresh = argmax(target_logits(matched), n_vocab);
        impl.n_matched += matched;

        // --- keep: emit the round (capped), drop everything past it ---
        // Round tokens: head at pos, matched drafts, new fresh token.
        const int32_t fresh_pos = pos + 1 + matched;
        int32_t last_kept = pos - 1;
        bool stop = produced >= n_predict;
        auto keep = [&](int32_t token, int32_t at) {
            if (stop) {
                return;
            }
            if (model.is_eog(token)) {
                // Like sequential generation: end markers are never emitted
                // and never kept in memory.
                llama_memory_seq_rm(impl.target_memory(), seq, at, -1);
                stop = true;
                return;
            }
            emit(token);
            ++produced;
            last_kept = at;
            stop = produced >= n_predict;
        };
        if (head_is_new) {
            keep(head, pos);
            head_is_new = false;
        }
        for (int32_t i = 0; i < matched; ++i) {
            keep(drafts[static_cast<std::size_t>(i)], pos + 1 + i);
        }
        keep(fresh, fresh_pos);
        // The fresh token is only sampled, never decoded: it takes its
        // position in the next round's batch. Evict whatever the verify
        // decode left there (a rejected draft, or nothing for a bonus).
        // A capped round instead drops everything past its last token.
        if (last_kept == fresh_pos) {
            llama_memory_seq_rm(impl.target_memory(), seq, fresh_pos, -1);
        } else {
            llama_memory_seq_rm(impl.target_memory(), seq, last_kept + 1, -1);
        }
        if (stop) {
            return produced;
        }
        // Hidden state just before the fresh token primes the next round.
        impl.read_pending(matched);
        pos = fresh_pos;
    }
    return produced;
}

int Speculative::n_mtp_layers() const {
    return impl_->n_mtp_layers;
}

int64_t Speculative::drafted() const {
    return impl_->n_drafted;
}

int64_t Speculative::matched() const {
    return impl_->n_matched;
}

double Speculative::acceptance_rate() const {
    if (impl_->n_drafted == 0) {
        return -1.0;
    }
    return static_cast<double>(impl_->n_matched) / static_cast<double>(impl_->n_drafted);
}

}  // namespace ggml_extractor
