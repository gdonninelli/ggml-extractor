#include "ggml_extractor/multimodal.hpp"

#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <stdexcept>
#include <utility>

#include "ggml_extractor/model.hpp"
#include "ggml_extractor/session.hpp"

namespace ggml_extractor {

struct Multimodal::Impl {
    std::shared_ptr<Model> model;
    mtmd_context* mctx = nullptr;

    ~Impl() {
        if (mctx != nullptr) {
            mtmd_free(mctx);
        }
    }
};

Multimodal::Multimodal(std::shared_ptr<Model> model, const std::string& mmproj_path,
                       const MultimodalOptions& options)
    : impl_(new Impl()) {
    if (!model) {
        throw std::runtime_error("Multimodal requires a loaded Model");
    }
    impl_->model = std::move(model);

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = options.use_gpu;
    params.print_timings = options.print_timings;
    params.n_threads = options.n_threads;
    params.image_min_tokens = options.image_min_tokens;
    params.image_max_tokens = options.image_max_tokens;

    impl_->mctx = mtmd_init_from_file(mmproj_path.c_str(), impl_->model->handle(), params);
    if (impl_->mctx == nullptr) {
        throw std::runtime_error("failed to load multimodal projector: " + mmproj_path);
    }
}

Multimodal::~Multimodal() = default;

bool Multimodal::supports_vision() const {
    return mtmd_support_vision(impl_->mctx);
}

bool Multimodal::supports_audio() const {
    return mtmd_support_audio(impl_->mctx);
}

std::string Multimodal::marker() const {
    const char* marker = mtmd_get_marker(impl_->mctx);
    return marker != nullptr ? std::string(marker) : std::string();
}

void Multimodal::eval(Session& session, int32_t seq, const std::string& prompt,
                      const std::vector<std::string>& media_paths, bool logits_last) {
    mtmd::bitmaps bitmaps;
    for (const std::string& path : media_paths) {
        mtmd_helper_bitmap_wrapper loaded = mtmd_helper_bitmap_init_from_file(
            impl_->mctx, path.c_str(), /*placeholder=*/false);
        if (loaded.video_ctx != nullptr) {
            mtmd_helper_video_free(loaded.video_ctx);
            if (loaded.bitmap != nullptr) {
                mtmd_bitmap_free(loaded.bitmap);
            }
            throw std::runtime_error("video input is not supported here: " + path);
        }
        if (loaded.bitmap == nullptr) {
            throw std::runtime_error("failed to load media file: " + path);
        }
        bitmaps.entries.emplace_back(loaded.bitmap);
    }

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) {
        throw std::runtime_error("failed to allocate mtmd input chunks");
    }

    const llama_pos n_past = session.n_past(seq);

    mtmd_input_text text;
    text.text = prompt.c_str();
    text.text_len = prompt.size();
    // BOS belongs to the start of a conversation only — adding it again mid
    // sequence would inject a second one on every follow-up turn.
    text.add_special = n_past == 0;
    text.parse_special = true;

    std::vector<const mtmd_bitmap*> handles = bitmaps.c_ptr();
    const int32_t tokenized = mtmd_tokenize(impl_->mctx, chunks.ptr.get(), &text,
                                            handles.data(), handles.size());
    if (tokenized == 1) {
        throw std::runtime_error("prompt must contain exactly one '" + marker() +
                                 "' per media file (" + std::to_string(handles.size()) +
                                 " given)");
    }
    if (tokenized != 0) {
        throw std::runtime_error("media preprocessing failed (mtmd_tokenize returned " +
                                 std::to_string(tokenized) + ")");
    }

    llama_pos new_n_past = n_past;
    const int32_t evaluated = mtmd_helper_eval_chunks(
        impl_->mctx, session.handle(), chunks.ptr.get(), n_past, seq,
        static_cast<int32_t>(session.n_batch()), logits_last, &new_n_past);
    if (evaluated != 0) {
        throw std::runtime_error("failed to evaluate multimodal prompt (code " +
                                 std::to_string(evaluated) + ")");
    }
    llama_synchronize(session.handle());
}

}  // namespace ggml_extractor
